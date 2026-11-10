import re, os, time
import requests
import json
import torch
import argparse
import importlib.util
import gradio as gr
from difflib import SequenceMatcher
from audio_cache import (
    asset_identity, cache_key, file_identity, valid_audio_cache,
    publish_audio_manifest, store_audio_cache,
)
from rvc_cache import find_model_index, inference_recipe, model_path
from functools import wraps
from threading import RLock
import tempfile

_CONVERT_LOCK = RLock()


def _serial_conversion(function):
    # Both Gradio endpoints share model state. A second request must not
    # replace the model or shared separator input during the current song.
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _CONVERT_LOCK:
            return function(*args, **kwargs)
    return wrapped

parser = argparse.ArgumentParser()
parser.add_argument(
    '--is_nohalf', action='store_true'
)
a = parser.parse_args()
is_half=not a.is_nohalf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}
pattern = r'//www\.bilibili\.com/video[^"]*'
models=[]
index=[]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_API_BASE = "http://127.0.0.1:7897"

# 自适应变调模块（与本项目同级的 adaptive_pitch.py，支持 RMVPE GPU 加速；
# 需要 rmvpe.pt + infer/rmvpe.py 在同级目录，缺失时自动回退 pyin CPU）
ADAPTIVE_PITCH_PY = os.path.normpath(os.path.join(SCRIPT_DIR, "adaptive_pitch.py"))

def _load_adaptive_pitch_module():
    """加载 adaptive_pitch.py 模块，缓存到全局避免重复读取。"""
    global _adaptive_mod
    if '_adaptive_mod' in globals() and _adaptive_mod is not None:
        return _adaptive_mod
    spec = importlib.util.spec_from_file_location("adaptive_pitch", ADAPTIVE_PITCH_PY)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    globals()["_adaptive_mod"] = mod
    return mod

def calc_adaptive_key_shift(vocal_path, low_note="A3", high_note="A5", use_stem_direct=False):
    """调用自适应变调脚本，根据纯人声计算最优升降调（半音）。
    任何异常都回退到 0，保证主流程不中断。
    use_stem_direct=True 时：直接对该声部 wav 跑 RMVPE，不走 UVR 缓存。"""
    if not os.path.isfile(vocal_path):
        print(f"[自适应变调] 人声文件不存在: {vocal_path}，跳过")
        return 0
    try:
        mod = _load_adaptive_pitch_module()
        if mod is None:
            print(f"[自适应变调] 无法加载 {ADAPTIVE_PITCH_PY}，跳过")
            return 0
        if use_stem_direct:
            # 直接对声部 wav 文件跑 F0（双通道专用）
            midi, weights = mod.extract_f0(vocal_path, device=device)
        else:
            try:
                midi, weights = mod.extract_f0(vocal_path, device=device)
            except TypeError:
                # 兼容旧版 adaptive_pitch.py（没有 device 参数）
                midi, weights = mod.extract_f0(vocal_path)
        r = mod.recommend_transpose(midi, weights, low_note, high_note)
        t = int(r.get("transpose", 0))
        print(f"[自适应变调] 舒适区 {low_note}-{high_note} → 推荐变调 {t:+d} 半音 (覆盖率 {r.get('coverage', 0)*100:.1f}%)")
        return t
    except Exception as e:
        print(f"[自适应变调] 计算失败，回退为 0: {e}")
        return 0

def optimize_pitch_shift(key_shift):
    if key_shift > 6:
        return key_shift - 12
    elif key_shift < -6:
        return key_shift + 12
    else:
        return key_shift


def _infer_one_stem(stem_wav_path, key_shift, model_name, index_path, f0_method, song_key, tag):
    """Reuse a validated dry stem; leave it intact for later remixes."""
    directory = os.path.join(os.path.dirname(os.path.abspath(stem_wav_path)), "rvc_infer")
    return _cached_rvc_inference(stem_wav_path, key_shift, model_name, f0_method, directory, tag)


def get_direct_audio(url):
    """Download an audio file from a direct link (mp3/wav/flac/m4a/ogg etc.)."""
    print(f"Starting download from direct link: {url}")
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://music.163.com/",
        "Accept": "*/*"
    }
    try:
        res = requests.get(url, headers=req_headers, timeout=60, stream=True, allow_redirects=True)
        if res.status_code != 200:
            raise gr.Error(f"Direct link download failed, HTTP {res.status_code} ({url})")
        content = res.content
        if len(content) < 10240:
            raise gr.Error("The returned content of the direct link is too small, likely not a valid audio file.")
        print(f"[Download successful] Direct link audio source (size: {len(content) // 1024} KB)")
        return content
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Direct link download exception: {e}")

def cookie_to_dict(cookie_string):
    cookie_dict = {}
    for item in cookie_string.split(';'):
        try:
            key, value = item.strip().split('=', 1)
            cookie_dict[key] = value
        except ValueError:
            pass
    return cookie_dict
    
current_model_index = ""

def change_model(model):
  global current_model_index
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_change_voice", json={
      "data": [
        model,
        0.33,
        0.33,
    ]}, timeout=10).json()

    if response.get("error") or not response.get("data"):
      raise RuntimeError(f"RVC 模型加载失败：{response}")
    current_model_index = ""
    for item in response.get("data", []):
      if isinstance(item, dict) and item.get("value") and str(item.get("value")).endswith(".index"):
        current_model_index = item["value"]
        break

    if current_model_index:
      print(f"模型已切换为: {model}，自动匹配索引: {current_model_index}")
    else:
      print(f"模型已切换为: {model}")
    return f"✅ 成功切换到模型: {model}"
  except Exception as e:
    print(f"切换模型失败: {e}")
    return f"❌ 切换模型失败: {e}"

def show_model():
  global models, index
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_refresh", json={
      "data": []
    }, timeout=10).json()

    data = response.get("data", [])
    if len(data) > 0 and isinstance(data[0], dict):
      models = data[0].get("choices", [])
    elif len(data) > 0 and isinstance(data[0], list):
      models = data[0]
    else:
      models = []

    if len(data) > 1 and isinstance(data[1], dict):
      index = data[1].get("choices", [])
    elif len(data) > 1 and isinstance(data[1], list):
      index = data[1]
    else:
      index = []

    print(f"已加载 {len(models)} 个模型")
    return models
  except Exception as e:
    print(f"获取模型列表失败: {e}")
    return []

def find_index(model):
    return find_model_index(model)


from uvr5.vr import AudioPre
from msst_backend import MSST_SELECTION, separate_msst_audio, resolve_model_asset
# 双通道（男/女声部分离）模块：缺失/异常时自动回退整轨，不中断主流程
try:
    import mf_separate
except Exception as _mf_import_err:
    mf_separate = None
    print(f"[双通道] mf_separate 模块不可用（{_mf_import_err}），双通道将自动回退整轨")
weight_uvr5_root = "uvr5/uvr_model"

def download_file_openxlab(url, destination):
  if os.path.exists(destination):
    return
  directory = os.path.dirname(destination)
  if directory:
    os.makedirs(directory, exist_ok=True)
  while True:
    try:
      response = requests.get(url, timeout=60)
      if response.status_code == 200:
        with open(destination, "wb") as file:
          file.write(response.content)
        return
      print(f"下载 VR-HP5 模型失败，HTTP {response.status_code}")
    except Exception as error:
      print(f"下载 VR-HP5 模型失败: {error}")
    time.sleep(5)

download_file_openxlab(
  "https://modelscope.cn/api/v1/models/CCYellowStar/5_HP-Karaoke-UVR/repo?Revision=master&FilePath=5_HP-Karaoke-UVR.pth",
  os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"),
)

# 初始化 VR 模型
func = AudioPre
pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"),
  device=device,
  is_half=is_half,
)

from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.effects import compress_dynamic_range
from pydub.effects import normalize
from pedalboard import Pedalboard, Compressor, Reverb
from scipy.signal import firwin, lfilter, iirfilter
import os
import numpy as np
import librosa
import soundfile
import gradio as gr
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = np.hanning

def wwy_downloader(filename, uvr_choice, direct_link):
    if uvr_choice in ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]:
        split_model = "MSST-BS-RoFormer"
        print(">>> 使用 MSST 串联管线 (becruily_deux + 去混响) 进行分离...")
    else:
        split_model = "UVR-HP5"
        print("🚀 使用 VR-HP5 模型进行分离...")

    # 只通过直链下载
    if not direct_link or not str(direct_link).startswith(("http://", "https://")):
        raise gr.Error("缺少有效的音频下载直链！")
    audio_content = get_direct_audio(str(direct_link).strip())
    temp_prefixed_path = "rvc_" + filename.strip() + ".wav"
    with open(temp_prefixed_path, mode="wb") as f:
        f.write(audio_content)

    # 下载到的本地 wav → 交给本地分离函数（与上传入口共用）
    vocal_path, inst_path = separate_local_file(temp_prefixed_path, filename, uvr_choice)

    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    return vocal_path, inst_path


def _separation_recipe(source_identity, mode):
    global _VR_LOADED_ASSETS, pre_fun_hp5
    if mode in ("MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"):
        assets = []
        for key in ("becruily_deux", "dereverb_echo"):
            _, weight, config = resolve_model_asset(key)
            assets.extend([asset_identity(weight), asset_identity(config)])
        code_root = os.path.join(SCRIPT_DIR, "msst", "inference")
        code = [asset_identity(os.path.join(SCRIPT_DIR, "msst_backend.py"))]
    else:
        assets = [asset_identity(pre_fun_hp5.model_path), asset_identity(
            os.path.join(SCRIPT_DIR, "uvr5/lib/lib_v5/modelparams/4band_v2.json"))]
        if globals().get("_VR_LOADED_ASSETS") != assets:
            if globals().get("_VR_LOADED_ASSETS") is not None:
                pre_fun_hp5 = AudioPre(agg=10, model_path=pre_fun_hp5.model_path,
                                      device=device, is_half=is_half)
            _VR_LOADED_ASSETS = assets
        code_root = os.path.join(SCRIPT_DIR, "uvr5")
        code = []
    for root, _, files in os.walk(code_root):
        for name in sorted(files):
            if name.endswith(".py"):
                code.append(asset_identity(os.path.join(root, name)))
    return {"recipe_version": 1, "source": source_identity, "mode": mode,
            "assets": assets, "code": code, "max_duration_ms": 300000,
            "half": is_half, "device": device,
            "vr_options": pre_fun_hp5.data if mode == "VR-HP5" else None}


def separate_local_file(local_audio_path, filename, uvr_choice):
    """对一个本地音频文件做 UVR 人声/伴奏分离，返回 (vocal_path, inst_path)。

    供直链下载（wwy_downloader）与本地上传（convert）两条入口共用。
    local_audio_path: 已落地的音频文件路径（wav/mp3/flac 等，pydub 可读）
    filename: 缓存子目录名（song_key）
    uvr_choice: 分离模型选择
    """
    if uvr_choice in ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]:
        split_model = "MSST-BS-RoFormer"
        print(">>> 使用 MSST 串联管线 (becruily_deux + 去混响) 进行分离...")
    else:
        split_model = "UVR-HP5"
        print("🚀 使用 VR-HP5 模型进行分离...")

    audio_orig = AudioSegment.from_file(local_audio_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"原始音频时长: {duration_minutes:.2f} 分钟")
    if duration_minutes > 5:
        print("⚠️ 音频超过5分钟，正在截取前5分钟...")
        audio_orig = audio_orig[:300000]

    os.makedirs(os.path.join(SCRIPT_DIR, "temp"), exist_ok=True)
    uvr_input_path = os.path.join(SCRIPT_DIR, "temp", filename.strip() + ".wav")
    audio_orig.export(uvr_input_path, format="wav")

    output_dir = f"./output/{split_model}/{filename}/"
    os.makedirs(output_dir, exist_ok=True)

    # ========== 执行分离 ==========
    if uvr_choice in ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]:
        vocal_path, inst_path = separate_msst_audio(
            uvr_input_path,
            output_dir,
            model_selection=uvr_choice,
        )
    else:
        # VR 模型分离
        pre_fun_hp5._path_audio_(uvr_input_path, output_dir, output_dir, "wav")
        vocal_path = f"{output_dir}/vocal_{filename}.wav_10.wav"
        inst_path = f"{output_dir}/instrument_{filename}.wav_10.wav"
    # ==============================

    if os.path.isfile(uvr_input_path):
        os.remove(uvr_input_path)

    return vocal_path, inst_path



@_serial_conversion
def convert(direct_link, key_shift, vocal_vol, inst_vol, model_dropdown, reverb_intensity, delay_intensity, uvr_choice="VR-HP5 (默认)", f0_method="rmvpe", adaptive_shift=False, comfort_low="A3", comfort_high="A5", a_capella=False, dual_channel=False, upload_file=None, harmony_model="", harmony_comfort_low="A3", harmony_comfort_high="A5", harmony_vol=0):

  # 双通道模式：硬编码走 MSST 串联管线（becruily_deux + 去混响）
  # 原因：MF 模型（Chorus BS-Roformer）需要在"干声足够干净"的前提下才能正确分离男/女声部，
  # VR-HP5 的干声残留伴奏/混响偏多，喂给 MF 会退化；MSST 串联管线干声干净，分离才正常。
  # 老通道（dual_channel=False）不受影响，仍用用户选择的 UVR 模型。
  if dual_channel:
    if uvr_choice not in ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]:
      print("[双通道] 强制切换分离模型：VR-HP5 → MSST 串联管线（becruily_deux + 去混响）")
      uvr_choice = "MSST 串联管线 (becruily_deux + 去混响)"

  # 双通道专属：伴唱参数预处理（dual_channel 关闭时这些参数完全不影响老通道）
  # harmony_model 留空 = 跟主唱（model_dropdown）用同一音色
  # harmony_comfort_low/high 留空 = 跟主唱舒适区一致
  # harmony_vol = 伴唱（男声轨）相对主唱的音量偏移 dB，默认 0
  if dual_channel:
    har_model = (harmony_model or "").strip()
    if not har_model:
      har_model = model_dropdown  # 留空 → 跟主唱同音色
    har_low = (harmony_comfort_low or "").strip() or comfort_low
    har_high = (harmony_comfort_high or "").strip() or comfort_high
  else:
    har_model = model_dropdown
    har_low = comfort_low
    har_high = comfort_high

  # 映射选择到内部标识
  if uvr_choice in ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]:
      current_uvr_mode = MSST_SELECTION
      split_model_dir = "MSST-BS-RoFormer"
  else:
      current_uvr_mode = "VR-HP5"
      split_model_dir = "UVR-HP5"

  # 音源入口：本地上传文件 优先于 直链
  upload_path = None
  if upload_file:
    # gradio gr.File 在 filepath 模式下返回 str 路径；兼容 list / dict 形式
    if isinstance(upload_file, (list, tuple)):
      upload_file = upload_file[0] if upload_file else None
    if isinstance(upload_file, dict):
      upload_file = upload_file.get("name")
    if isinstance(upload_file, str) and upload_file.strip():
      upload_path = os.path.abspath(upload_file.strip())
      if not os.path.isfile(upload_path):
        upload_path = None  # 路径不存在则回退直链

  if upload_path:
    song_key = re.sub(r'[\\/:*?"<>|]', '', os.path.splitext(os.path.basename(upload_path))[0])[:32] or "audio"
    source_identity = file_identity(upload_path)
    direct_link = None
    print(f"📂 处理本地上传文件: {os.path.basename(upload_path)} | 使用分离模型: {current_uvr_mode}")
  else:
    if not direct_link or not direct_link.strip():
      raise gr.Error("请填写音频下载直链，或上传一个音频文件！")
    direct_link = direct_link.strip()
    if not direct_link.startswith(("http://", "https://")):
      raise gr.Error("直链必须以 http:// 或 https:// 开头！")
    # A URL can serve different audio tomorrow. Download first and identify
    # its bytes instead of treating the URL as an immutable audio source.
    content = get_direct_audio(direct_link)
    import hashlib
    source_identity = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    song_key = "download"
    os.makedirs(os.path.join(SCRIPT_DIR, "temp"), exist_ok=True)
    fd, upload_path = tempfile.mkstemp(prefix="download-", suffix=".audio", dir=os.path.join(SCRIPT_DIR, "temp"))
    with os.fdopen(fd, "wb") as stream:
      stream.write(content)

  # Cache namespaces are content-addressed; equal names alone never imply a hit.
  output_label = (re.sub(r'[\\/:*?"<>|]', '', os.path.splitext(os.path.basename(upload_path))[0])[:80]
                  if direct_link is None else "download")
  song_key += "_" + source_identity["sha256"][:16]
  split_payload = _separation_recipe(source_identity, current_uvr_mode)
  split_key = cache_key(split_payload)[:16]
  song_key += "_" + split_key
  split_dir = os.path.join(SCRIPT_DIR, "output", split_model_dir, song_key)
  os.makedirs(split_dir, exist_ok=True)
  audio_rvc_path = os.path.join(split_dir, "rvc_input.wav")
  if split_model_dir == "MSST-BS-RoFormer":
    expected_vocal_path = os.path.join(split_dir, song_key + "_vocals.wav")
    expected_inst_path = os.path.join(split_dir, song_key + "_other.wav")
  else:
    expected_vocal_path = os.path.join(split_dir, "vocal_" + song_key + ".wav_10.wav")
    expected_inst_path = os.path.join(split_dir, "instrument_" + song_key + ".wav_10.wav")
  manifest = os.path.join(split_dir, ".split-manifest.json")
  try:
    if valid_audio_cache(manifest, split_payload, [expected_vocal_path, expected_inst_path]):
      print(f"✅ 人声／伴奏缓存命中 ({current_uvr_mode})")
      vocal_path, inst_path = expected_vocal_path, expected_inst_path
    else:
      if os.path.isfile(manifest):
        os.remove(manifest)
      print("📂 未找到有效缓存，开始分离")
      vocal_path, inst_path = separate_local_file(upload_path, song_key, current_uvr_mode)
      publish_audio_manifest(manifest, split_payload, [vocal_path, inst_path])
  finally:
    if direct_link and os.path.isfile(upload_path):
      os.remove(upload_path)
  audio_rvc, sr_src = librosa.load(vocal_path, sr=44100, mono=True)
  soundfile.write(audio_rvc_path, audio_rvc, sr_src)

  # === 双通道模式：MF 提纯男/女声部，各自自适应变调 + 各自 7897 推理，再合成 ===
  if dual_channel:
    print("\n" + "=" * 65)
    print(">>> 双通道模式：MSST 串联管线干声 → MF 分离男/女声部 → 各自自适应变调")
    print(">>> 保守串行：声部1 F0 → 声部1 7897推理 → 声部2 F0 → 声部2 7897推理")
    print(">>> 完成后合成两声部 + 伴奏，再走老通道混响/延迟/混音输出")
    print("=" * 65)

    # 1) MF 分离（失败自动回退整轨干声，不中断主流程）
    mf_ok = False
    if mf_separate is not None:
      try:
        mf_out_dir = f"./output/{split_model_dir}/{song_key}/mf"
        os.makedirs(mf_out_dir, exist_ok=True)
        male_audio, female_audio = mf_separate.separate_male_female(
            vocal_path, mf_out_dir, device="auto", sr=44100
        )
        if male_audio is not None and female_audio is not None:
          mf_ok = True
          male_wav, female_wav = mf_separate.get_stem_wav_paths(vocal_path, mf_out_dir)
          print(f"[双通道] MF 分离成功：male→{os.path.basename(male_wav)}, female→{os.path.basename(female_wav)}")
        else:
          print("[双通道] MF 分离返回空（模型输出键异常），回退整轨干声")
      except Exception as e:
        print(f"[双通道] MF 分离异常（{e}），回退整轨干声")

    if not mf_ok:
      print("[双通道] ⚠️ MF 模块不可用或失败，本次回退为整轨推理（老通道）")
      # 显式调用老通道单轨推理兜底，保证 data 被赋值
      data = _run_single_track_infer(
        audio_rvc_path, key_shift, model_dropdown, f0_method,
        vocal_path, adaptive_shift, comfort_low, comfort_high, inst_path
      )
      # 跳过后面的双通道逻辑（mf_ok 为 False，下面 if dual_channel 分支不会执行）
      dual_channel = False

    # 双通道下两声部可能各自变调，伴奏段用伴唱（男声）的 shift 作为代表
    dual_key_shift = 0
    if dual_channel and mf_ok:
      # 2) 伴奏先放一边（后续合成用）
      print(f"[双通道] 伴奏 {os.path.basename(inst_path)} 已备好，待合成")
      print(f"[双通道] 主唱 → 女声轨（音色 {model_dropdown}，舒适区 {comfort_low}-{comfort_high}）")
      print(f"[双通道] 伴唱 → 男声轨（音色 {har_model}，舒适区 {har_low}-{har_high}，音量 {harmony_vol:+d}dB）")

      # 3) 保守串行：先主唱（女声）再伴唱（男声）
      # 每轨各自的音色 + 舒适区 + 音量
      stem_configs = [
        ("主唱(女声)", female_wav, model_dropdown, comfort_low, comfort_high, 0),
        ("伴唱(男声)", male_wav, har_model, har_low, har_high, harmony_vol),
      ]
      male_inferred = None
      female_inferred = None
      sr_stem = 44100
      for tag, stem_wav, stem_model, c_low, c_high, stem_vol_db in stem_configs:
        print(f"\n[双通道] 处理 {tag} 声部…")
        # 3a) 该声部自适应变调（用各自的舒适区；手动 key_shift 为 0 才启用）
        stem_shift = key_shift
        if adaptive_shift and not key_shift:
          computed = calc_adaptive_key_shift(stem_wav, c_low, c_high, use_stem_direct=True)
          if computed != 0:
            print(f"[双通道] {tag} 自适应变调：{key_shift} → {computed:+d} 半音（舒适区 {c_low}-{c_high}）")
            stem_shift = computed
          else:
            print(f"[双通道] {tag} 自适应变调：推荐值 0，保持不变调")
        stem_shift = int(max(-12, min(12, stem_shift)))
        # 伴唱（男声）shift 作为伴奏变调代表值（伴奏通常跟男声低音区走）
        if tag.startswith("伴唱"):
          dual_key_shift = stem_shift

        # 3b) 该声部 7897 推理（各自的音色；blocking；等上一个完成才发下一个——天然不阻塞）
        switch_model(stem_model)
        stem_index = find_index(stem_model)
        try:
          stem_out_wav = _infer_one_stem(stem_wav, stem_shift, stem_model, stem_index, f0_method, song_key, tag)
        except gr.Error:
          raise
        # 把该声部输出读入内存（两声部都要保留用于合成），并按各自的音量 dB 调整
        _stem_out, _sr_stem = librosa.load(stem_out_wav, sr=None, mono=True)
        # 伴唱（男声轨）按 harmony_vol dB 缩放；主唱（女声轨）保持原幅（跟 vocal_vol 走）
        if tag.startswith("伴唱") and harmony_vol != 0:
          _gain = 10.0 ** (float(harmony_vol) / 20.0)
          _stem_out = _stem_out * _gain
        if tag.startswith("主唱"):
          female_inferred = _stem_out
          sr_stem = _sr_stem
        else:
          male_inferred = _stem_out

      # 4) 合成两声部 → 写入一个合并人声 wav（44100 mono，与老通道一致）
      print("\n[双通道] 合成男声 + 女声两路推理结果…")
      # 两声部经 RVC 后采样数可能差几十~几百个采样（去静音端点不同），
      # 截断到较短的再相加，避免 broadcast 报错
      n_min = min(len(male_inferred), len(female_inferred))
      male_inferred = male_inferred[:n_min]
      female_inferred = female_inferred[:n_min]
      mixed = male_inferred + female_inferred
      # 归一化到峰值 0.95，避免叠加削波
      if mixed.max() > 0:
        mixed = mixed / mixed.max() * 0.95
      mixed_wav_path = os.path.join("temp", f"{song_key}_mf_mixed.wav")
      os.makedirs("./temp", exist_ok=True)
      import wave
      _pcm = (np.clip(mixed, -1, 1) * 32767).astype(np.int16)
      with wave.open(mixed_wav_path, "wb") as _w:
        _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(int(sr_stem))
        _w.writeframes(_pcm.tobytes())
      print(f"[双通道] 合成完成 → {mixed_wav_path}（{len(mixed)/sr_stem:.1f}s）")

      # 5) 把合成结果送老通道的音效处理（混响/延迟/压缩/响度）+ 伴奏合成
      #    让 data 指向合成 wav，复用下方 if data: 的 pedalboard 流程（不重复造轮子）
      data = mixed_wav_path
      # 双通道：整轨 key_shift 用伴唱（男声）代表值，供伴奏变调段使用
      key_shift = dual_key_shift
      print(f"[双通道] 已落入老通道音效处理段（伴奏变调代表值 {dual_key_shift:+d}，混响/延迟/混音 + 伴奏合成）")
  else:
    # === 老通道：整轨单轨推理（原逻辑不变）===
    data = _run_single_track_infer(
      audio_rvc_path, key_shift, model_dropdown, f0_method,
      vocal_path, adaptive_shift, comfort_low, comfort_high, inst_path
    )

  if data:
      print("🎛️ 开始处理音频")
      os.makedirs("./temp", exist_ok=True)

      audio_data, sr = librosa.load(data, sr=None, mono=False)

      if audio_data.ndim == 1:
          audio_data = audio_data.reshape(1, -1)

      from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter, PitchShift, Delay

      room_size_map =  (0.15,          0.40,          0.90)
      wet_level_map =  (0.10,          0.25,          0.45)

      if reverb_intensity <= 4:
          percent = reverb_intensity / 4.0
          room_size_val = room_size_map[0] + (room_size_map[1] - room_size_map[0]) * percent
          wet_level_val = wet_level_map[0] + (wet_level_map[1] - wet_level_map[0]) * percent
      else:
          percent = (reverb_intensity - 4) / 6.0 
          room_size_val = room_size_map[1] + (room_size_map[2] - room_size_map[1]) * percent
          wet_level_val = wet_level_map[1] + (wet_level_map[2] - wet_level_map[1]) * percent

      dry_level_val = 1.0 - wet_level_val

      print(f"🎤 混响设置: 强度 {reverb_intensity}/10 => 房间大小={room_size_val:.2f}, 湿润度={wet_level_val:.2f}")
    
      effects = [
          HighpassFilter(cutoff_frequency_hz=80),
          PeakFilter(cutoff_frequency_hz=200, gain_db=1.5, q=0.7),
          PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.0),
          PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
          LowpassFilter(cutoff_frequency_hz=16000),
          Compressor(
              threshold_db=-18.0,
              ratio=4.0,
              attack_ms=5.0,
              release_ms=150.0
          ),
      ]
    
      if delay_intensity > 0:
          print("🎤 启用回声效果...")
          try:
              y_inst, sr_inst = librosa.load(inst_path, sr=None)
              tempo, _ = librosa.beat.beat_track(y=y_inst, sr=sr_inst)
              actual_tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo

              if actual_tempo > 0:
                  delay_seconds_val = (60.0 / actual_tempo) * 0.5 
              else:
                  delay_seconds_val = 0.5
          except Exception:
              delay_seconds_val = 0.5
            
          delay_mix_val = (delay_intensity / 10.0) * 0.35
          effects.append(Delay(delay_seconds=delay_seconds_val, feedback=0.25, mix=delay_mix_val))
    
      effects.append(Reverb(room_size=room_size_val, damping=0.4, wet_level=wet_level_val, dry_level=dry_level_val, width=0.8))
    
      board = Pedalboard(effects)
      processed = board(audio_data, sr)
      processed_int16 = (processed.T * 32768).astype(np.int16)
      processed_audio = AudioSegment(processed_int16.tobytes(), frame_rate=sr, sample_width=2, channels=processed.shape[0])
    
      audio_vocal_adjusted = processed_audio + vocal_vol
      normalized_audio = normalize(audio_vocal_adjusted, headroom=-1.0)
    
      # ===== 清唱模式：不合成伴奏，直接导出处理后的纯人声 =====
      if a_capella:
          print("🎤 清唱模式：跳过伴奏合成，直接导出纯人声")
          output_path = f"temp/{sanitize_filename(output_label)}-清唱.mp3"
          normalized_audio.export(output_path, format="MP3", bitrate="192k")
          _cleanup_mix_input(data)
          print(f"✅ 已导出(清唱): {output_path}")
          return output_path

      print("🎵 准备伴奏...")
      key_shift = optimize_pitch_shift(key_shift)
    
      if key_shift != 0 and abs(key_shift) != 12:
          print(f"🎹 正在将伴奏音高调整 {key_shift:+d} 半音...")
          try:
              y_inst, sr_inst = librosa.load(inst_path, sr=None)
              # 统一成 mono，绕开 libsndfile 写立体声 wav 的 "Format not recognised" 老 bug
              if y_inst.ndim == 2:
                  y_inst = y_inst.mean(axis=0)
              pitch_board = Pedalboard([PitchShift(semitones=key_shift)])
              y_shifted = pitch_board(y_inst, sr_inst)

              shifted_inst_path = f"temp/shifted_{song_key}_inst.wav"
              # 立体声 mono 混音后用 stdlib wave 写 16-bit PCM（soundfile 立体声写崩）
              import wave
              _pcm = (np.clip(y_shifted, -1, 1) * 32767).astype(np.int16)
              with wave.open(shifted_inst_path, "wb") as _w:
                  _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(int(sr_inst))
                  _w.writeframes(_pcm.tobytes())
              audio_inst = AudioSegment.from_file(shifted_inst_path, format="wav")
          except Exception as e:
              print(f"⚠️ 伴奏调整失败: {e}")
              audio_inst = AudioSegment.from_file(inst_path, format="wav")
      else:
          audio_inst = AudioSegment.from_file(inst_path, format="wav")

      audio_inst = audio_inst + inst_vol
      combined_audio = normalized_audio.overlay(audio_inst)

      print("💾 导出最终文件...")
      output_path = f"temp/{sanitize_filename(output_label)}-RVC-AI翻唱.mp3"
      combined_audio.export(output_path, format="MP3", bitrate="192k")
    
      _cleanup_mix_input(data)
    
      print(f"✅ 已导出: {output_path}")
      return output_path


def _run_single_track_infer(audio_rvc_path, key_shift, model_dropdown, f0_method,
                            vocal_path, adaptive_shift, comfort_low, comfort_high, inst_path):
    """Cache dry RVC output independently of downstream mix/effect settings."""
    if adaptive_shift and not key_shift:
        key_shift = calc_adaptive_key_shift(vocal_path, comfort_low, comfort_high)
    directory = os.path.join(os.path.dirname(os.path.abspath(vocal_path)), "rvc_infer")
    return _cached_rvc_inference(audio_rvc_path, key_shift, model_dropdown, f0_method, directory)


def _cached_rvc_inference(audio_path, key_shift, model, f0_method, directory, tag="single"):
    key_shift = int(max(-12, min(12, key_shift)))
    f0_method = f0_method if f0_method in ("pm", "rmvpe", "fcpe") else "rmvpe"
    audio_path = os.path.abspath(audio_path)

    def locate():
        index_path = find_index(model)
        payload = inference_recipe(audio_path, model, index_path, key_shift, f0_method, tag)
        path = os.path.join(directory, cache_key(payload)[:32] + ".wav")
        return index_path, payload, path

    index_path, payload, cached_wav = locate()
    if valid_audio_cache(cached_wav + ".json", payload, [cached_wav]):
        print(f"✅ RVC 干声缓存命中：{os.path.basename(cached_wav)}（model={model}, shift={key_shift:+d}, f0={f0_method}）")
        return cached_wav

    status = switch_model(model)
    if not str(status).startswith("✅"):
        raise gr.Error(str(status))
    # Loading can discover a different index from the backend. Key the actual
    # index used, and check again before doing any expensive inference.
    index_path, payload, cached_wav = locate()
    if valid_audio_cache(cached_wav + ".json", payload, [cached_wav]):
        print(f"✅ RVC 干声缓存命中：{os.path.basename(cached_wav)}")
        return cached_wav

    def request_inference(legacy=False):
        if legacy:
            data = [0, audio_path, key_shift, None, f0_method, "", index_path,
                    0.75, 3, 0, 0.25, 0.33]
        else:
            data = [0, {"name": audio_path, "data": None, "is_file": True},
                    key_shift, f0_method, index_path, 0.75, 0, 0.25, 0.33]
        try:
            response = requests.post(f"{RVC_API_BASE}/run/infer_convert", json={"data": data},
                                     timeout=globals().get("TIMEOUT", 240))
            response.raise_for_status()
            result = response.json()
            outputs = result.get("data", [])
            if not result.get("error") and len(outputs) > 1 and isinstance(outputs[1], dict):
                path = outputs[1].get("name") or outputs[1].get("path")
                if path and os.path.isfile(path):
                    return path
        except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
            print(f"[RVC] 推理请求失败：{exc}")
        return None

    print(f"🎤 RVC 推理中...（model={model}, shift={key_shift:+d}, f0={f0_method}）")
    output_wav = request_inference()
    if output_wav is None and globals().get("_LAST_SWITCH_SKIPPED"):
        print("[模型] 常驻模型推理失败，强制重新加载后重试一次…")
        status = switch_model(model, force=True)
        if not str(status).startswith("✅"):
            raise gr.Error(str(status))
        index_path, payload, cached_wav = locate()
        output_wav = request_inference()
    if output_wav is None:
        output_wav = request_inference(legacy=True)
    if output_wav is None:
        raise gr.Error("RVC 推理失败：没有返回可读取的音频，请检查 7897 服务日志。")
    store_audio_cache(output_wav, cached_wav, payload)
    print(f"💾 RVC 干声已缓存：{os.path.basename(cached_wav)}")
    return cached_wav


def _cleanup_mix_input(path):
    # Only the merged two-stem working file belongs to the mix stage.
    # A path returned by _cached_rvc_inference is a retained cache asset.
    temporary_root = os.path.normcase(os.path.abspath(os.path.join(SCRIPT_DIR, "temp")))
    actual = os.path.normcase(os.path.abspath(path))
    if os.path.dirname(actual) == temporary_root and actual.endswith("_mf_mixed.wav"):
        if os.path.isfile(path):
            os.remove(path)


def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '', filename)

def refresh_models():
    models_list = show_model()
    if models_list:
        return gr.Dropdown(choices=models_list, value=models_list[0] if models_list else None)
    else:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")

# 7897 侧当前已加载的模型名 / 上一次切换是否走了"跳过"路径
_LOADED_MODEL = None
_LOADED_MODEL_ASSET = None
_LAST_SWITCH_SKIPPED = False


@_serial_conversion
def switch_model(model_name, force=False):
    """切换到指定模型；若 7897 已常驻同名模型则跳过重复加载（省 2~8s）。

    7897 侧 infer_change_voice 每次都会重新读 .pth + index，
    连续翻唱同一音色时完全没必要。若复用的常驻模型实际不可用
    （例如 7897 中途重启过），调用方会清空 _LOADED_MODEL 重试一次。
    force=True 时无条件重新加载（「切换模型」按钮走这条，用于和 7897 手动换模型后重新对齐）。
    """
    global _LOADED_MODEL, _LOADED_MODEL_ASSET, _LAST_SWITCH_SKIPPED
    if not model_name or model_name == "无可用模型":
        return "❌ 请先选择一个有效的模型"
    try:
        model_asset = asset_identity(model_path(model_name))
    except OSError as exc:
        return f"❌ 无法读取 RVC 模型（可用 RVC_PROJECT_ROOT 指定项目目录）：{exc}"
    if not force and _LOADED_MODEL == model_name and _LOADED_MODEL_ASSET == model_asset:
        _LAST_SWITCH_SKIPPED = True
        print(f"[模型] {model_name} 已常驻 7897，跳过重复加载")
        return f"✅ 当前已是模型: {model_name}"
    msg = change_model(model_name)
    _LAST_SWITCH_SKIPPED = False
    _LOADED_MODEL = model_name if str(msg).startswith("✅") else None
    _LOADED_MODEL_ASSET = model_asset if _LOADED_MODEL else None
    return msg

    
app = gr.Blocks()

with app:
  gr.Markdown("# <center>RVC一键翻唱、重磅更新！</center>")
  gr.Markdown("## 自动分离人声翻唱并合并，自动混音！</center>")
  
  with gr.Row():
    with gr.Column():
      with gr.Row():
        model_dropdown = gr.Dropdown(label="选择AI模型", choices=[], value=None, info="请先点击刷新加载模型列表")
        refresh_btn = gr.Button("🔄 刷新", size="sm")
        switch_btn = gr.Button("✨ 切换模型", size="sm", variant="primary")
      with gr.Row(visible=False):
          models_json = gr.JSON()
          get_models_btn = gr.Button("获取模型列表", visible=False)
          get_models_btn.click(show_model, outputs=models_json)
      with gr.Row():
        model_status = gr.Textbox(label="模型状态", value="请选择模型", interactive=False)
      
      with gr.Row():
        inp1 = gr.Textbox(label="请填写音乐下载直链", placeholder="https://example.com/xxx.mp3", info="填写可直接下载音频的链接（mp3/wav/flac/m4a等）")
      
      # ========== 音源入口：本地上传文件（优先于直链） ==========
      gr.Markdown("或上传音频文件（mp3/wav/flac/m4a/ogg，最长5分钟）；上传文件优先于直链，按音频内容和配置校验缓存，同名不同文件分别处理")
      inp_upload = gr.File(
          label="上传音频文件",
          file_count="single",
          file_types=["audio", "video"]
      )
      # ======================================
      
      # ========== 新增：选择分离模型 ==========
      with gr.Row():
          uvr_model_choice = gr.Radio(
              label="人声分离模型 (UVR)",
              choices=["VR-HP5 (默认)", "MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"],
              value="VR-HP5 (默认)",
              info="MSST 串联管线：Step 1 becruily_deux 提取人声伴奏 -> Step 2 dereverb_echo 提取纯净干声"
          )
      # ======================================

      # ========== 音高提取算法（F0 method） ==========
      with gr.Row():
          f0_method_choice = gr.Dropdown(
              label="音高提取算法 (F0)",
              choices=["rmvpe", "pm", "fcpe"],
              value="rmvpe",
              info="rmvpe=默认，精度最高；pm=旧版；fcpe=全卷积，适合无 RMVPE 的轻量模型"
          )
      # ======================================

      # ========== 自适应变调 ==========
      with gr.Row():
          adaptive_shift_choice = gr.Checkbox(
              label="自适应变调",
              value=False,
              info="开启后，若升降调为 0，将根据纯人声音域自动计算最优升降调；非 0 时以手动值为准"
          )
          comfort_low_input = gr.Textbox(
              label="舒适区低音",
              value="A3",
              placeholder="A3",
              info="如 A3 / C4 / G3"
          )
          comfort_high_input = gr.Textbox(
              label="舒适区高音",
              value="A5",
              placeholder="A5",
              info="如 A5 / C6 / G4"
          )
      # ======================================

      # ========== 清唱模式 ==========
      with gr.Row():
          a_capella_choice = gr.Checkbox(
              label="清唱模式（不合成伴奏）",
              value=False,
              info="开启后，推理完人声直接输出纯人声（保留混响/延迟/压缩/响度效果），不再叠加伴奏轨道"
          )
      # ======================================

      # ========== 双通道（男/女声部分离）模式 ==========
      with gr.Row():
          dual_channel_choice = gr.Checkbox(
              label="双声道模式（男/女声部分离）",
              value=False,
              info="开启后：MSST 串联管线分离出干净干声 → MF 分离主唱(女声)/伴唱(男声)声部 → 各自音色+自适应变调+各自 RVC 推理 → 合成两声部 + 伴奏 → 老通道混响/延迟输出"
          )
      # ======================================

      # ========== 双通道专属：伴唱音色/音域/音量（仅双通道开启时生效） ==========
      with gr.Row():
          harmony_model_input = gr.Textbox(
              label="伴唱音色（男声轨推理用）",
              value="",
              placeholder="留空 = 跟主唱同一音色",
              info="填写 RVC 模型名（如 wuxiV2.pth）；留空则伴唱跟主唱用同一音色"
          )
          harmony_comfort_low_input = gr.Textbox(
              label="伴唱舒适区低音",
              value="A3",
              placeholder="A3",
              info="伴唱（男声）自适应变调的舒适区下限，如 A2 / C3 / G2"
          )
          harmony_comfort_high_input = gr.Textbox(
              label="伴唱舒适区高音",
              value="A4",
              placeholder="A4",
              info="伴唱（男声）自适应变调的舒适区上限，如 A4 / C5 / G4"
          )
          harmony_vol_input = gr.Slider(
              minimum=-20, maximum=10, value=0, step=1,
              label="伴唱音量 (dB)",
              info="伴唱（男声）轨相对主唱的音量偏移，默认 0，建议 -6（伴唱轻一点）"
          )
      # ======================================

      with gr.Row():
        inp5 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
        inp6 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节人声音量，默认为0")
        inp7 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节伴奏音量，默认为0")
      with gr.Row():
        inp_reverb = gr.Slider(minimum=0, maximum=10, value=4, step=0.5, label="混响强度", info="0为干声，4为默认值，10为宏大混响")
        inp_delay = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="回声(延迟)效果", info="0为关闭，数值越大回声越明显")
      btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
    
    with gr.Column():
      out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False,streaming=True,)

  refresh_btn.click(refresh_models, outputs=model_dropdown,api_name=None)
  switch_btn.click(lambda m: switch_model(m, force=True), inputs=model_dropdown, outputs=model_status)
  
  # 绑定参数：直链/升降调/人声音量/伴奏音量/模型/混响/回声/UVR模型/F0算法/自适应变调/低音/高音/清唱/双通道/上传文件/伴唱音色/伴唱低音/伴唱高音/伴唱音量
  btn.click(convert, [inp1, inp5, inp6, inp7, model_dropdown, inp_reverb, inp_delay, uvr_model_choice, f0_method_choice, adaptive_shift_choice, comfort_low_input, comfort_high_input, a_capella_choice, dual_channel_choice, inp_upload, harmony_model_input, harmony_comfort_low_input, harmony_comfort_high_input, harmony_vol_input], out, api_name="None")
  
  api_model_name = gr.Textbox(visible=False)
  api_output = gr.Audio(visible=False)
  gr.Button("API Convert", visible=False).click(
      convert,
      inputs=[inp1, inp5, inp6, inp7, api_model_name, inp_reverb, inp_delay, uvr_model_choice, f0_method_choice, adaptive_shift_choice, comfort_low_input, comfort_high_input, a_capella_choice, dual_channel_choice, inp_upload, harmony_model_input, harmony_comfort_low_input, harmony_comfort_high_input, harmony_vol_input],
      outputs=[api_output],
      api_name="convert"
  )
  gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")


if __name__ == "__main__":
    print("正在初始化并加载模型列表...")
    initial_models = show_model()
    if initial_models:
        print(f"成功加载 {len(initial_models)} 个模型")
    else:
        print("⚠️ 警告: 未能加载模型列表，请确保RVC服务正在运行")

    app.queue(max_size=40, api_open=False)
    app.launch(server_name="0.0.0.0", share=False, show_error=True)
