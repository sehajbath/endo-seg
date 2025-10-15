"""
Configuration loading and management utilities
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    Later configs override earlier ones

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    merged = {}

    for config in configs:
        merged = _deep_update(merged, config)

    return merged


def _deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update nested dictionary"""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def save_config(config: Dict[str, Any], save_path: str, format: str = 'yaml'):
    """
    Save configuration to file

    Args:
        config: Configuration dictionary
        save_path: Path to save config
        format: File format ('yaml' or 'json')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'yaml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif format == 'json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved configuration to: {save_path}")


def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load all configuration files from directory and merge them

    Args:
        config_dir: Directory containing config files

    Returns:
        Merged configuration
    """
    config_dir = Path(config_dir)

    # Load configs in order
    config_files = [
        'config.yaml',
        'data_config.yaml',
        'model_config.yaml',
        'training_config.yaml'
    ]

    configs = []
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            configs.append(load_yaml_config(str(config_path)))
        else:
            logger.warning(f"Config file not found: {config_path}")

    # Merge all configs
    merged_config = merge_configs(*configs)

    logger.info(f"Loaded and merged {len(configs)} configuration files")
    return merged_config


def override_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values from command-line arguments

    Args:
        config: Base configuration
        args: Command-line arguments (can use dot notation for nested keys)

    Returns:
        Updated configuration
    """
    for key, value in args.items():
        if value is None:
            continue

        # Handle nested keys (e.g., 'training.batch_size')
        if '.' in key:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value

    logger.info("Configuration overridden from arguments")
    return config


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
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
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary"""
        training_config = config_dict.get('training', {})
        return cls(**{k: v for k, v in training_config.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str = "swin_unetr"
    in_channels: int = 1
    num_classes: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        model_config = config_dict.get('model', {})
        input_config = model_config.get('input', {})
        output_config = model_config.get('output', {})

        return cls(
            name=model_config.get('name', 'swin_unetr'),
            in_channels=input_config.get('in_channels', 1),
            num_classes=output_config.get('num_classes', 4)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class ConfigManager:
    """Manage configuration throughout training"""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize config manager

        Args:
            config_path: Path to config file (YAML or JSON)
            config_dict: Configuration dictionary (if not loading from file)
        """
        if config_path is not None:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = load_yaml_config(config_path)
            elif config_path.endswith('.json'):
                self.config = load_json_config(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Key to retrieve (can use dots for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key: Key to set (can use dots for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def save(self, save_path: Optional[str] = None, format: str = 'yaml'):
        """
        Save configuration to file

        Args:
            save_path: Path to save (uses original path if None)
            format: File format
        """
        if save_path is None:
            save_path = self.config_path

        if save_path is None:
            raise ValueError("No save path specified")

        save_config(self.config, save_path, format)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"ConfigManager({self.config_path})"


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        'paths.data_root',
        'model.name',
        'training.batch_size',
        'training.epochs'
    ]

    for field in required_fields:
        keys = field.split('.')
        current = config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise ValueError(f"Missing required config field: {field}")
            current = current[key]

    logger.info("Configuration validation passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test loading config
    try:
        config = load_yaml_config('../configs/config.yaml')
        print("Loaded main config")
        print(f"Project name: {config.get('project_name')}")
        print(f"Batch size: {config.get('training', {}).get('batch_size')}")

        # Test ConfigManager
        config_mgr = ConfigManager(config_dict=config)
        print(f"\nUsing ConfigManager:")
        print(f"Batch size: {config_mgr.get('training.batch_size')}")
        print(f"Model name: {config_mgr.get('model.name')}")

        # Test TrainingConfig dataclass
        train_config = TrainingConfig.from_dict(config)
        print(f"\nTraining config dataclass:")
        print(f"  Batch size: {train_config.batch_size}")
        print(f"  Learning rate: {train_config.learning_rate}")
        print(f"  Epochs: {train_config.epochs}")

        print("\n✓ Configuration utilities test successful!")

    except FileNotFoundError:
        print("Config file not found (expected if not in project root)")