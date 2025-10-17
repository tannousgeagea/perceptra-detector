"""
Configuration management utilities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import os
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    return config


def apply_env_overrides(config: Dict[str, Any], prefix: str = "PERCEPTRA_") -> Dict[str, Any]:
    """
    Override config values with environment variables.
    
    Environment variables should be in format: PERCEPTRA_SECTION_KEY
    Example: PERCEPTRA_DETECTOR_DEVICE=cuda
    
    Args:
        config: Configuration dictionary
        prefix: Environment variable prefix
        
    Returns:
        Updated configuration
    """
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # Parse key path
        path = key[len(prefix):].lower().split('_')
        
        # Navigate to nested dict
        current = config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set value
        final_key = path[-1]
        
        # Try to parse value
        try:
            # Try as JSON
            import json
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Use as string
            parsed_value = value
        
        current[final_key] = parsed_value
        logger.info(f"Override from env: {'.'.join(path)} = {parsed_value}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {output_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "detector": {
            "model_path": "yolov8n.pt",
            "backend": None,  # Auto-detect
            "device": None,  # Auto-detect
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "auto_warmup": False,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "enable_cors": True,
            "models": {},
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }


class Config:
    """
    Configuration manager class.
    
    Example:
        >>> config = Config.from_file("config.yaml")
        >>> detector = Detector(**config.detector)
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary."""
        self._config = config_dict
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load from file."""
        config_dict = load_config(config_path)
        return cls(config_dict)
    
    @classmethod
    def from_default(cls) -> 'Config':
        """Create from default config."""
        return cls(get_default_config())
    
    def __getattr__(self, name: str) -> Any:
        """Get config section as attribute."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._config.get(name)
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    def save(self, output_path: str) -> None:
        """Save to file."""
        save_config(self._config, output_path)