"""男/女声部分离（BS-Roformer Chorus Male/Female 模型）集成模块。

在 UVR 人声/伴奏分离之后、自适应变调计算之前，对 UVR 输出的干声
做进一步的"男/女声部提纯"。复用 7860 现成的 MSST 框架
（msst/inference/msst_infer.py 的 MSSeparator，model_type=bs_roformer），
零新依赖、零额外模型代码——只需 ckpt + config 两个文件。

模型权重：Sucial/Chorus_Male_Female_BS_Roformer（BS-Roformer 架构，
训练自 opencpop + m4singer 中文歌曲，ep_267 SDR≈24.13）。
注意：该模型只能分离"合唱（男女同时唱）"，男女交替（一段一段）时
无法分离，输出会退化为整轨人声——检测方需自行判断。

回退语义：
  - separate_male_female() 任何异常（模型缺失/加载失败/推理失败/显存不足）
    都返回 (None, None)，调用方（app_rvc.convert）应直接使用原始 UVR 干声
    继续流程，不中断主流程。
  - 成功时返回 (male_audio, female_audio)，shape (channels, samples)，
    与 MSST separate() 输出一致，可直接喂给后续 F0/RVC 流程。
"""
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch

# MSST 框架路径（与本模块同级）
MSST_ROOT = Path(__file__).resolve().parent / "msst"
if str(MSST_ROOT) not in __import__("sys").path:
    import sys
    sys.path.insert(0, str(MSST_ROOT))

from inference.msst_infer import MSSeparator  # noqa: E402

# 模型资产目录：msst/pretrain/male_female_models/
_MF_MODEL_DIR = MSST_ROOT / "pretrain" / "male_female_models"
# 默认用 SDR 更高的 ep_267；若缺失可回退 ep_146
_CKPT_CANDIDATES = [
    _MF_MODEL_DIR / "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
    _MF_MODEL_DIR / "model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt",
]
_CONFIG_PATH = _MF_MODEL_DIR / "config_chorus_male_female_bs_roformer.yaml"

# 进程内缓存：(model_key, device) -> MSSeparator
_MF_SEPARATORS: dict = {}
_MF_LOCK = None  # 延迟创建


def _get_lock():
    global _MF_LOCK
    if _MF_LOCK is None:
        import threading
        _MF_LOCK = threading.RLock()
    return _MF_LOCK


def _write_wav_pcm16(audio, sr, out_path):
    """用标准库 wave 写 16-bit PCM wav（统一写 mono，避免立体声交错头写入溢出）。

    runtime 的 soundfile 对立体声 wav 有 "Format not recognised" 的 libsndfile
    后端缺陷，wave 模块不依赖 libsndfile，稳定。人声提纯取能量即可，mono 足够。
    """
    import wave
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=0)  # 多声道 → mono
    a = a.reshape(-1)
    if a.max() > 1.0:
        a = a / a.max()
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def _resolve_mf_assets():
    """定位 ckpt + config。返回 (model_type, ckpt_path, config_path)。"""
    ckpt = next((p for p in _CKPT_CANDIDATES if p.is_file()), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"未找到男/女分离模型权重，期望: {list(_CKPT_CANDIDATES)}\n"
            f"请下载 Sucial/Chorus_Male_Female_BS_Roformer 的 ckpt 到该目录"
        )
    cfg = _CONFIG_PATH
    if not cfg.is_file():
        raise FileNotFoundError(
            f"未找到男/女分离模型配置: {cfg}\n"
            f"请下载 config_chorus_male_female_bs_roformer.yaml"
        )
    return "bs_roformer", ckpt, cfg


def _get_mf_separator(device: str, output_dir: Path) -> MSSeparator:
    with _get_lock():
        key = (device, str(output_dir))
        sep = _MF_SEPARATORS.get(key)
        if sep is not None:
            return sep
        model_type, ckpt, cfg = _resolve_mf_assets()
        print(f"[MF分离] 加载 BS-Roformer 男女分离模型: {ckpt.name} @ {device}", flush=True)
        t0 = time.perf_counter()
        sep = MSSeparator(
            model_type=model_type,
            config_path=str(cfg),
            model_path=str(ckpt),
            device=device,
            output_format="wav",
            store_dirs=str(output_dir),
            inference_params={
                "batch_size": None,
                "num_overlap": None,
                "chunk_size": None,
                "normalize": None,
            },
        )
        _MF_SEPARATORS[key] = sep
        print(f"[MF分离] 模型加载完成，耗时 {time.perf_counter() - t0:.1f}s", flush=True)
        return sep


def separate_male_female(vocal_path, output_dir, device="auto", sr=44100):
    """对 UVR 输出的人声做男/女声部提纯。

    参数:
      vocal_path: UVR 人声 wav 路径（44100Hz，mono/stereo 均可）
      output_dir: 输出目录（放 male/female 轨 + 临时文件）
      device: "auto" / "cuda:0" / "cpu"
    返回:
      成功: (male_audio, female_audio)，np.ndarray (channels, samples) @ sr
      失败: (None, None)——调用方应回退到原始 UVR 人声，不中断主流程
    """
    import traceback
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) 解析 device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        # 0) 缓存复用：输出目录已存在 male/female wav 且非空 → 直接返回，跳过 MF 推理
        stem = Path(str(vocal_path)).stem
        cached_male = out / f"{stem}_male.wav"
        cached_female = out / f"{stem}_female.wav"
        if cached_male.is_file() and cached_female.is_file() and cached_male.stat().st_size > 44 and cached_female.stat().st_size > 44:
            t0 = time.perf_counter()
            male, male_sr = librosa.load(str(cached_male), sr=sr, mono=True)
            female, female_sr = librosa.load(str(cached_female), sr=sr, mono=True)
            print(f"[MF分离] ✅ 缓存命中，跳过 MF 推理：{cached_male.name} / {cached_female.name}"
                  f"（耗时 {time.perf_counter()-t0:.1f}s）", flush=True)
            return male, female
        # 2) 读入 UVR 人声（保持 stereo；模型 stereo=true 要求 2ch）
        t0 = time.perf_counter()
        mix, _ = librosa.load(str(vocal_path), sr=sr, mono=False)
        print(f"[MF分离] 读入人声 {os.path.basename(str(vocal_path))}: "
              f"{mix.shape}, 耗时 {time.perf_counter() - t0:.1f}s", flush=True)

        # 3) 加载/复用分离器
        sep = _get_mf_separator(device, out)

        # 4) 推理
        t1 = time.perf_counter()
        results = sep.separate(mix)
        print(f"[MF分离] 推理完成，耗时 {time.perf_counter() - t1:.1f}s", flush=True)

        # 5) 提取 male / female 轨（键名按 config.training.instruments = [male, female]）
        norm = {str(k).lower(): v for k, v in results.items()}
        male = norm.get("male")
        female = norm.get("female")

        if male is None or female is None:
            # 键名异常：打印实际键便于排查，视为失败
            print(f"[MF分离] 输出键异常: {list(results.keys())}，"
                  f"期望 ['male','female']，回退原人声", flush=True)
            return None, None

        # MSST separate() 对 stereo 模型返回 (samples, channels) 布局；
        # 统一转成 (channels, samples) 供调用方使用
        def _to_ch_first(a):
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:
                return a.reshape(1, -1)
            if a.shape[0] > a.shape[1]:
                return a.T  # (samples, ch) -> (ch, samples)
            return a
        male = _to_ch_first(male)
        female = _to_ch_first(female)

        # 6) 写出 wav（供人工试听/排查）
        # 注意：runtime 的 soundfile 对【立体声】wav 有 "Format not recognised" 的
        # libsndfile 后端缺陷（mono/flac 正常），改用 wave 模块写 16-bit PCM
        male_path = out / f"{Path(str(vocal_path)).stem}_male.wav"
        female_path = out / f"{Path(str(vocal_path)).stem}_female.wav"
        _write_wav_pcm16(male, sr, male_path)
        _write_wav_pcm16(female, sr, female_path)

        print(f"[MF分离] 成功: male → {male_path.name}, "
              f"female → {female_path.name}", flush=True)
        return male, female
    except Exception as e:
        print(f"[MF分离] 失败，回退原始 UVR 人声: {e}", flush=True)
        traceback.print_exc()
        return None, None


def get_stem_wav_paths(vocal_path, output_dir):
    """返回 (male_wav, female_wav) 的绝对路径（separate_male_female 写出的）。

    若文件不存在返回 (None, None)。供双通道推理侧直接读盘，避免内存传递。
    与 separate_male_female 内的命名规则保持一致。
    """
    out = Path(output_dir).resolve()
    stem = Path(str(vocal_path)).stem
    m = out / f"{stem}_male.wav"
    f = out / f"{stem}_female.wav"
    if m.is_file() and f.is_file():
        return str(m), str(f)
    return None, None
