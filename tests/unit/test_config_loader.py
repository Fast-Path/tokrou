"""Unit tests for configuration loading."""

import pytest
from backend.core.config_loader import (
    load_profile,
    load_pricing,
    load_token_estimation,
    load_simulation,
    list_available_profiles
)
from backend.core.config_validator import ConfigValidationError


def test_load_baseline_profile():
    """Test loading the baseline profile."""
    profile = load_profile('baseline')
    
    assert profile['name'] == 'baseline'
    assert 'query_distribution' in profile
    assert 'complexity_distribution' in profile
    assert 'routing_accuracy' in profile
    assert 'queries_per_day' in profile


def test_load_conservative_profile():
    """Test loading the conservative profile."""
    profile = load_profile('conservative')
    
    assert profile['name'] == 'conservative'
    assert profile['routing_accuracy'] == 0.70


def test_load_optimistic_profile():
    """Test loading the optimistic profile."""
    profile = load_profile('optimistic')
    
    assert profile['name'] == 'optimistic'
    assert profile['routing_accuracy'] == 0.95


def test_load_nonexistent_profile():
    """Test loading a profile that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_profile('nonexistent')


def test_load_pricing():
    """Test loading pricing configuration."""
    pricing = load_pricing()
    
    assert 'models' in pricing
    assert 'gemini' in pricing['models']
    assert 'coder' in pricing['models']
    assert 'grok' in pricing['models']
    assert 'classifier' in pricing['models']
    
    # Check each model has input and output pricing
    for model in ['gemini', 'coder', 'grok', 'classifier']:
        assert 'input' in pricing['models'][model]
        assert 'output' in pricing['models'][model]


def test_load_token_estimation():
    """Test loading token estimation configuration."""
    config = load_token_estimation()
    
    assert 'system_prompts' in config
    assert 'input_ranges' in config
    assert 'output_multipliers' in config
    assert 'wrong_model_penalty' in config


def test_load_simulation():
    """Test loading simulation configuration."""
    config = load_simulation()
    
    assert 'runs' in config
    assert 'days_to_simulate' in config
    assert 'sensitivity_analysis' in config


def test_list_available_profiles():
    """Test listing available profiles."""
    profiles = list_available_profiles()
    
    assert 'baseline' in profiles
    assert 'conservative' in profiles
    assert 'optimistic' in profiles
    assert len(profiles) >= 3
