"""
modules/mnemosyne/config.py

Self-contained config for Mnemosyne when running inside Hestia.
No external config.json required — all defaults are sensible.
Override any value by passing kwargs to MnemosyneConfig().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Root of the Mnemosyne module (modules/mnemosyne/)
_MODULE_DIR = Path(__file__).parent          # hestia/modules/mnemosyne/
# Hestia project root (modules/mnemosyne -> modules -> hestia/)
_HESTIA_ROOT = _MODULE_DIR.parent.parent     # hestia/


@dataclass
class MnemosyneConfig:
    db_path: str = str(_HESTIA_ROOT / "data" / "mnemosyne" / "mnemosyne.db")
    chroma_dir: str = str(_HESTIA_ROOT / "data" / "mnemosyne" / "chroma_db")
    summarise_every_n: int = 20  # compress after every 20 interactions
    max_facts: int = 500
    max_goals: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        # Ensure the parent directory for db_path exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        # Ensure chroma_dir exists
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)

# Module-level singleton config
_config: Optional[MnemosyneConfig] = None

def get_config() -> MnemosyneConfig:
    global _config
    if _config is None:
        _config = MnemosyneConfig()
    return _config

def set_config(config: MnemosyneConfig) -> None:
    global _config
    _config = config
