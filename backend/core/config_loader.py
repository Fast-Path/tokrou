"""Configuration loading utilities."""

import json
from typing import Dict, Any
from pathlib import Path

from .config_validator import (
    validate_profile_config,
    validate_pricing_config,
    validate_token_estimation_config,
    ConfigValidationError
)


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Assume config directory is at project root
    return Path(__file__).parent.parent.parent / "config"


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary containing JSON contents
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing JSON file {filepath}: {e.msg}", e.doc, e.pos)


def load_profile(profile_name: str) -> Dict[str, Any]:
    """
    Load a usage profile configuration.
    
    Args:
        profile_name: Name of the profile (without .json extension)
    
    Returns:
        Dictionary containing profile configuration
    
    Raises:
        FileNotFoundError: If profile doesn't exist
        ConfigValidationError: If profile is invalid
    """
    config_dir = get_config_dir()
    profile_path = config_dir / "profiles" / f"{profile_name}.json"
    
    config = load_json_file(profile_path)
    
    # Validate profile
    errors = validate_profile_config(config)
    if errors:
        raise ConfigValidationError(
            f"Invalid profile '{profile_name}':\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config


def load_pricing() -> Dict[str, Any]:
    """
    Load pricing configuration.
    
    Returns:
        Dictionary containing pricing configuration
    
    Raises:
        FileNotFoundError: If pricing.json doesn't exist
        ConfigValidationError: If pricing config is invalid
    """
    config_dir = get_config_dir()
    pricing_path = config_dir / "pricing.json"
    
    config = load_json_file(pricing_path)
    
    # Validate pricing
    errors = validate_pricing_config(config)
    if errors:
        raise ConfigValidationError(
            "Invalid pricing configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config


def load_token_estimation() -> Dict[str, Any]:
    """
    Load token estimation configuration.
    
    Returns:
        Dictionary containing token estimation configuration
    
    Raises:
        FileNotFoundError: If token_estimation.json doesn't exist
        ConfigValidationError: If token estimation config is invalid
    """
    config_dir = get_config_dir()
    token_path = config_dir / "token_estimation.json"
    
    config = load_json_file(token_path)
    
    # Validate token estimation config
    errors = validate_token_estimation_config(config)
    if errors:
        raise ConfigValidationError(
            "Invalid token estimation configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config


def load_simulation() -> Dict[str, Any]:
    """
    Load simulation configuration.
    
    Returns:
        Dictionary containing simulation configuration
    
    Raises:
        FileNotFoundError: If simulation.json doesn't exist
    """
    config_dir = get_config_dir()
    simulation_path = config_dir / "simulation.json"
    
    return load_json_file(simulation_path)


def list_available_profiles() -> list[str]:
    """
    List all available profile names.
    
    Returns:
        List of profile names (without .json extension)
    """
    config_dir = get_config_dir()
    profiles_dir = config_dir / "profiles"
    
    if not profiles_dir.exists():
        return []
    
    profiles = []
    for file in profiles_dir.glob("*.json"):
        profiles.append(file.stem)
    
    return sorted(profiles)
