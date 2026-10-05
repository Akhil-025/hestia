"""
modules/iris/config.py

Config loader for Iris (media module).
Reads from global Hestia YAML config.
Falls back to safe defaults if missing.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────
_MODULE_DIR = Path(__file__).parent
_HESTIA_ROOT = _MODULE_DIR.parent.parent


# ── Config Dataclass ───────────────────────────────────
@dataclass
class IrisConfig:
    db_path: str
    source_dir: str
    output_dir: str
    cache_dir: str
    chroma_dir: str
    batch_size: int = 50
    max_workers: int = 4
    use_gpu: bool = False
    # Combined Hamming distance (average-hash + phash, each 64-bit with the
    # imagehash defaults used in ingestion.py) below which two images count
    # as near-duplicates. 0 = identical hashes. ~10-14 catches re-saves,
    # recompressions, and minor resizes/crops without false-positiving on
    # genuinely different photos; raise it to be stricter about what counts
    # as "duplicate enough to skip", lower it to only catch closer matches.
    perceptual_hash_threshold: int = 12
    # backlog #80: warn (never silently stop) before ingesting a folder
    # that would push total ingested media past this many bytes. None (the
    # default) disables the guard entirely — most people don't want a
    # quota until they've hit a real storage problem once.
    storage_quota_bytes: Optional[int] = None
    # backlog #71: optional relevance cut-offs for semantic search. CLIP always
    # returns the nearest photos, however poor a match they are; these drop the
    # poor ones. Cosine *distance*, lower = closer. None (default) = off, since
    # sensible values depend on your library and have to be tuned (see
    # modules/iris/embeddings.py, filter_hits).
    semantic_max_distance: Optional[float] = None
    semantic_relative_margin: Optional[float] = None
    # backlog #75: frames sampled per video
    video_frames: int = 4
    # backlog #72: face grouping. OFF by default: face embeddings are biometric.
    faces_enabled: bool = False
    face_detector_model: str = ""
    face_recognizer_model: str = ""
    face_match_threshold: float = 0.45
    face_min_cluster_size: int = 2
    face_min_size: int = 40
    # backlog #77: camera object detection. OFF by default: it opens the webcam.
    camera_enabled: bool = False
    camera_index: int = 0
    detector_model: str = "yolov8n.pt"
    detector_confidence: float = 0.4

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


# ── Singleton ──────────────────────────────────────────
_config: Optional[IrisConfig] = None


# ── Loader ─────────────────────────────────────────────
def get_config(path: str = "config/laptop_config.yaml") -> IrisConfig:
    global _config

    if _config is not None:
        return _config

    # Load YAML safely
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    iris_cfg = cfg.get("iris", {})

    # ── Defaults ────────────────────────────────────────
    default_base = _HESTIA_ROOT / "data" / "iris"

    _config = IrisConfig(
        db_path=iris_cfg.get("db_path", str(default_base / "iris.db")),
        source_dir=iris_cfg.get("source_dir", ""),
        output_dir=iris_cfg.get("output_dir", str(default_base / "organized")),
        cache_dir=iris_cfg.get("cache_dir", str(default_base / "cache")),
        chroma_dir=iris_cfg.get("chroma_dir", str(default_base / "chroma_db")),
        batch_size=iris_cfg.get("batch_size", 50),
        max_workers=iris_cfg.get("max_workers", 4),
        use_gpu=iris_cfg.get("use_gpu", False),
        perceptual_hash_threshold=iris_cfg.get("perceptual_hash_threshold", 12),
        storage_quota_bytes=iris_cfg.get("storage_quota_bytes"),
        semantic_max_distance=(iris_cfg.get("semantic") or {}).get("max_distance"),
        semantic_relative_margin=(iris_cfg.get("semantic") or {}).get("relative_margin"),
        video_frames=int((iris_cfg.get("video") or {}).get("frames", 4)),
        faces_enabled=bool((iris_cfg.get("faces") or {}).get("enabled", False)),
        face_detector_model=str((iris_cfg.get("faces") or {}).get("detector_model", "")),
        face_recognizer_model=str((iris_cfg.get("faces") or {}).get("recognizer_model", "")),
        face_match_threshold=float((iris_cfg.get("faces") or {}).get("match_threshold", 0.45)),
        face_min_cluster_size=int((iris_cfg.get("faces") or {}).get("min_cluster_size", 2)),
        face_min_size=int((iris_cfg.get("faces") or {}).get("min_face_size", 40)),
        camera_enabled=bool((iris_cfg.get("camera") or {}).get("enabled", False)),
        camera_index=int((iris_cfg.get("camera") or {}).get("index", 0)),
        detector_model=str((iris_cfg.get("camera") or {}).get("model", "yolov8n.pt")),
        detector_confidence=float((iris_cfg.get("camera") or {}).get("confidence", 0.4)),
    )

    return _config


# ── Override hook (optional) ───────────────────────────
def set_config(config: IrisConfig) -> None:
    global _config
    _config = config