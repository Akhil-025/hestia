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
    db_cfg = cfg.get("database", {})

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
    )

    return _config


# ── Override hook (optional) ───────────────────────────
def set_config(config: IrisConfig) -> None:
    global _config
    _config = config