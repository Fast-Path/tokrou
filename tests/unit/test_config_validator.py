"""Unit tests for configuration validation."""

import pytest
from backend.core.config_validator import (
    validate_distribution,
    validate_accuracy,
    validate_positive_integer,
    validate_profile_config,
    validate_pricing_config,
    validate_token_estimation_config
)


def test_validate_distribution_valid():
    """Test validating a valid distribution."""
    distribution = {'a': 0.3, 'b': 0.5, 'c': 0.2}
    errors = validate_distribution(distribution, 'test_dist')
    assert len(errors) == 0


def test_validate_distribution_invalid_sum():
    """Test validating a distribution that doesn't sum to 1.0."""
    distribution = {'a': 0.3, 'b': 0.5, 'c': 0.3}
    errors = validate_distribution(distribution, 'test_dist')
    assert len(errors) > 0
    assert 'sum to 1.0' in errors[0]


def test_validate_distribution_negative_value():
    """Test validating a distribution with negative values."""
    distribution = {'a': 0.5, 'b': -0.2, 'c': 0.7}
    errors = validate_distribution(distribution, 'test_dist')
    assert len(errors) > 0
    assert 'non-negative' in errors[0]


def test_validate_accuracy_valid():
    """Test validating valid accuracy values."""
    assert len(validate_accuracy(0.0, 'test')) == 0
    assert len(validate_accuracy(0.5, 'test')) == 0
    assert len(validate_accuracy(1.0, 'test')) == 0


def test_validate_accuracy_invalid():
    """Test validating invalid accuracy values."""
    errors = validate_accuracy(-0.1, 'test')
    assert len(errors) > 0
    
    errors = validate_accuracy(1.5, 'test')
    assert len(errors) > 0


def test_validate_positive_integer_valid():
    """Test validating valid positive integers."""
    assert len(validate_positive_integer(1, 'test')) == 0
    assert len(validate_positive_integer(1000, 'test')) == 0


def test_validate_positive_integer_invalid():
    """Test validating invalid positive integers."""
    errors = validate_positive_integer(0, 'test')
    assert len(errors) > 0
    
    errors = validate_positive_integer(-5, 'test')
    assert len(errors) > 0


def test_validate_profile_config_valid():
    """Test validating a valid profile configuration."""
    config = {
        'name': 'test',
        'description': 'Test profile',
        'query_distribution': {
            'visual': 0.2,
            'code': 0.3,
            'research': 0.5
        },
        'complexity_distribution': {
            'simple': 0.6,
            'medium': 0.3,
            'complex': 0.1
        },
        'routing_accuracy': 0.85,
        'delegation_accuracy': 0.90,
        'queries_per_day': 1000
    }
    
    errors = validate_profile_config(config)
    assert len(errors) == 0


def test_validate_profile_config_missing_field():
    """Test validating a profile with missing fields."""
    config = {
        'name': 'test',
        'description': 'Test profile'
    }
    
    errors = validate_profile_config(config)
    assert len(errors) > 0
    assert any('Missing required field' in e for e in errors)


def test_validate_profile_config_invalid_distribution():
    """Test validating a profile with invalid distribution."""
    config = {
        'name': 'test',
        'description': 'Test profile',
        'query_distribution': {
            'visual': 0.2,
            'code': 0.3,
            'research': 0.6  # Sums to 1.1
        },
        'complexity_distribution': {
            'simple': 0.6,
            'medium': 0.3,
            'complex': 0.1
        },
        'routing_accuracy': 0.85,
        'delegation_accuracy': 0.90,
        'queries_per_day': 1000
    }
    
    errors = validate_profile_config(config)
    assert len(errors) > 0
    assert any('sum to 1.0' in e for e in errors)


def test_validate_pricing_config_valid():
    """Test validating a valid pricing configuration."""
    config = {
        'models': {
            'gemini': {'input': 0.000001, 'output': 0.000003},
            'coder': {'input': 0.0000006, 'output': 0.0000006},
            'grok': {'input': 0.000005, 'output': 0.000015},
            'classifier': {'input': 0.0000001, 'output': 0.0000001}
        }
    }
    
    errors = validate_pricing_config(config)
    assert len(errors) == 0


def test_validate_pricing_config_missing_model():
    """Test validating pricing config with missing model."""
    config = {
        'models': {
            'gemini': {'input': 0.000001, 'output': 0.000003},
            'coder': {'input': 0.0000006, 'output': 0.0000006}
        }
    }
    
    errors = validate_pricing_config(config)
    assert len(errors) > 0


def test_validate_token_estimation_config_valid():
    """Test validating a valid token estimation configuration."""
    config = {
        'system_prompts': {
            'gemini': 150,
            'coder': 200,
            'grok': 180,
            'classifier': 50
        },
        'input_ranges': {
            'visual': {
                'simple': [100, 300],
                'medium': [300, 800],
                'complex': [800, 2000]
            },
            'code': {
                'simple': [50, 200],
                'medium': [200, 600],
                'complex': [600, 1500]
            },
            'research': {
                'simple': [100, 400],
                'medium': [400, 1000],
                'complex': [1000, 3000]
            }
        },
        'output_multipliers': {
            'simple': 0.5,
            'medium': 1.0,
            'complex': 1.5
        },
        'wrong_model_penalty': 1.3
    }
    
    errors = validate_token_estimation_config(config)
    assert len(errors) == 0
