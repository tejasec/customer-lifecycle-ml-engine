import os
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def load_yaml(file_path: Path) -> Dict[str, Any]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def get_config() -> Dict[str, Any]:
    config_path = ROOT_DIR / "configs" / "config.yaml"
    return load_yaml(config_path)

def get_model_config() -> Dict[str, Any]:
    model_config_path = ROOT_DIR / "configs" / "model_configs.yaml"
    return load_yaml(model_config_path)

def resolve_path(relative_or_absolute: str) -> Path:
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return ROOT_DIR / p
