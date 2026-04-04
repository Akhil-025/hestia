"""
modules/iris/config.py

Self-contained config for Iris image module in Hestia.
No external config.json required — all defaults are sensible.
Override any value by passing kwargs to IrisConfig().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Root of the Iris module (modules/iris/)
_MODULE_DIR = Path(__file__).parent
# Hestia project root (modules/iris -> modules -> hestia/)
_HESTIA_ROOT = _MODULE_DIR.parent.parent

@dataclass
class IrisConfig:
    db_path: str = str(_HESTIA_ROOT / "data" / "iris" / "iris.db")
    source_dir: str = "C:/Users/pilla/Pictures"
    output_dir: str = str(_HESTIA_ROOT / "data" / "iris" / "organized")
    cache_dir: str = str(_HESTIA_ROOT / "data" / "iris" / "cache")
    chroma_dir: str = str(_HESTIA_ROOT / "data" / "iris" / "chroma_db")
    batch_size: int = 50
    max_workers: int = 4
    use_gpu: bool = False

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

# Module-level singleton config
_config: Optional[IrisConfig] = None

def get_config() -> IrisConfig:
    global _config
    if _config is None:
        _config = IrisConfig()
    return _config

def set_config(config: IrisConfig) -> None:
    global _config
    _config = config
