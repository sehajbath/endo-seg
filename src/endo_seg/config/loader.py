"""
Configuration loading and management utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries (later configs override earlier ones)."""
    merged: Dict[str, Any] = {}
    for config in configs:
        merged = _deep_update(merged, config)
    return merged


def save_config(config: Dict[str, Any], save_path: str, fmt: str = "yaml") -> None:
    """Persist configuration to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "yaml":
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif fmt == "json":
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {fmt}")

    logger.info("Saved configuration to %s", save_path)


def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """Load and merge all known configuration files within a directory."""
    config_dir = Path(config_dir)
    config_files = [
        "config.yaml",
        "data_config.yaml",
        "model_config.yaml",
        "training_config.yaml",
    ]

    configs = []
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            configs.append(load_yaml_config(str(config_path)))
        else:
            logger.debug("Config file not found: %s", config_path)

    merged_config = merge_configs(*configs)
    logger.info("Loaded and merged %d configuration files", len(configs))
    return merged_config


def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration values using dot-notation keys."""
    for key, value in overrides.items():
        if value is None:
            continue

        current = config
        keys = key.split(".")
        for part in keys[:-1]:
            current = current.setdefault(part, {})
        current[keys[-1]] = value

    logger.info("Applied configuration overrides: %s", list(overrides.keys()))
    return config


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_workers: int = 4
    epochs: int = 200
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    early_stopping_patience: int = 30
    gradient_clip: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        training_config = config_dict.get("training", {})
        return cls(**{k: v for k, v in training_config.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    name: str = "swin_unetr"
    in_channels: int = 1
    num_classes: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        model_config = config_dict.get("model", {})
        input_config = model_config.get("input", {})
        output_config = model_config.get("output", {})

        return cls(
            name=model_config.get("name", "swin_unetr"),
            in_channels=input_config.get("in_channels", 1),
            num_classes=output_config.get("num_classes", 4),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """Runtime helper for accessing configuration via dot notation."""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        if config_path is not None:
            if config_path.endswith((".yaml", ".yml")):
                self.config = load_yaml_config(config_path)
            elif config_path.endswith(".json"):
                self.config = load_json_config(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        value = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        current = self.config
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def save(self, save_path: Optional[str] = None, fmt: str = "yaml") -> None:
        save_path = save_path or self.config_path
        if save_path is None:
            raise ValueError("No save path specified")
        save_config(self.config, save_path, fmt)

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def __repr__(self) -> str:
        return f"ConfigManager({self.config_path})"


def validate_config(config: Dict[str, Any]) -> bool:
    """Ensure configuration contains required fields."""
    required_fields = [
        "paths.data_root",
        "model.name",
        "training.batch_size",
        "training.epochs",
    ]

    for field in required_fields:
        current = config
        for key in field.split("."):
            if not isinstance(current, dict) or key not in current:
                raise ValueError(f"Missing required config field: {field}")
            current = current[key]

    logger.info("Configuration validation passed")
    return True


__all__ = [
    "ConfigManager",
    "ModelConfig",
    "TrainingConfig",
    "load_yaml_config",
    "load_json_config",
    "merge_configs",
    "save_config",
    "load_all_configs",
    "override_config",
    "validate_config",
]
