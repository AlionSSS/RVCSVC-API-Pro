"""Resolve local RVC assets before inference, including on a dry-vocal cache hit."""
import os
import re
from pathlib import Path

from audio_cache import asset_identity, file_identity

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "RVC20260718"


def project_root():
    return Path(os.environ.get("RVC_PROJECT_ROOT", str(DEFAULT_ROOT))).resolve()


def model_path(model):
    return project_root() / "assets" / "weights" / model


def find_model_index(model):
    # Match the backend's infer/vc/utils.py selection order without importing
    # its incompatible Python runtime or contacting/loading a model on a hit.
    stem = Path(model).stem
    experiment = re.sub(r"_e\d+_s\d+$", "", stem, flags=re.IGNORECASE).lower()
    candidates = []
    for rank, root in enumerate((project_root() / "assets/indices", project_root() / "logs")):
        if not root.is_dir():
            continue
        for path in root.rglob("*.index"):
            lower = path.stem.lower()
            if "trained" in lower:
                continue
            standard = (lower.startswith(experiment + "_added_")
                        or "_" + experiment + "_v1" in lower
                        or "_" + experiment + "_v2" in lower)
            if standard or stem.lower() in lower:
                score = (0 if standard else 1, rank, -path.stat().st_mtime, str(path).lower())
                candidates.append((score, str(path)))
    return min(candidates)[1] if candidates else ""


def inference_recipe(audio_path, model, index_path, shift, f0_method, tag="single"):
    root = project_root()
    # Inputs exclude reverb, mix volumes, output format and a_capella: those
    # are all downstream of the dry vocal and must be free to reuse it.
    code = [root / "configs/config.py"] + sorted((root / "infer").rglob("*.py"))
    assets = {"model": asset_identity(model_path(model))}
    for relative in ("assets/hubert_base/pytorch_model.bin", "assets/rmvpe/rmvpe.pt"):
        path = root / relative
        if path.is_file():
            assets[relative] = asset_identity(path)
    return {
        "recipe_version": 1,
        "input": file_identity(audio_path),
        "assets": assets,
        "index": asset_identity(index_path) if index_path else None,
        "backend_code": [asset_identity(path) for path in code],
        "segment_search": os.environ.get("RVC_SEGMENT_SEARCH", "prefix").lower(),
        "sid": 0, "shift": int(shift), "f0_method": f0_method,
        "index_rate": 0.75, "resample_sr": 0, "rms_mix_rate": 0.25,
        "protect": 0.33, "tag": tag,
    }
