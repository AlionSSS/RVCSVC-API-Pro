"""Content-addressed audio caches with validated, atomically published manifests."""
import hashlib
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import soundfile as sf

CACHE_VERSION = 1


def cache_key(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@lru_cache(maxsize=128)
def _file_digest(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    """Equal bytes in different upload directories have the same identity."""
    path = Path(path).resolve()
    before = path.stat()
    digest = _file_digest(str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise RuntimeError("File changed while checking cache identity: %s" % path)
    return {"sha256": digest, "bytes": before.st_size}


def asset_identity(path):
    path = Path(path).resolve()
    # Also track replacement/touch events so loaded model instances can be
    # invalidated even if a replacement happens to contain equal bytes.
    stat = path.stat()
    return {"path": str(path), "mtime_ns": stat.st_mtime_ns, **file_identity(path)}


def _audio_identity(path):
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("Empty or invalid cached audio: %s" % path)
    return {
        **file_identity(path), "frames": info.frames,
        "samplerate": info.samplerate, "channels": info.channels,
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def publish_audio_manifest(manifest, payload, paths):
    # Hash and validate every file before publishing the completion marker.
    audio = [_audio_identity(path) for path in paths]
    atomic_json(manifest, {"version": CACHE_VERSION, "key": cache_key(payload), "audio": audio})


def valid_audio_cache(manifest, payload, paths):
    try:
        with open(manifest, encoding="utf-8") as stream:
            recorded = json.load(stream)
        return (
            recorded.get("version") == CACHE_VERSION
            and recorded.get("key") == cache_key(payload)
            and recorded.get("audio") == [_audio_identity(path) for path in paths]
        )
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError):
        return False


def store_audio_cache(source, destination, payload):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".rvc-", suffix=".wav", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        _audio_identity(temporary)
        os.replace(temporary, destination)
        publish_audio_manifest(str(destination) + ".json", payload, [destination])
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return str(destination)
