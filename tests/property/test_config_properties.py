"""Property-based tests for configuration loading and validation."""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import yaml

import pytest
from hypothesis import given, strategies as st, assume

from backend.core.config_loader import (
    load_yaml_file,
    load_profile,
    load_pricing,
    load_token_estimation,
    load_simulation,
    get_config_dir
)
from backend.core.config_validator import ConfigValidationError
from backend.core.types import UsageProfile


# Strategies for generating valid configuration data

@st.composite
def valid_distribution(draw):
    """Generate a valid distribution that sums to 1.0."""
    # Generate 3 values that sum to 1.0
    values = draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3))
    total = sum(values)
    if total == 0:
        # Avoid division by zero
        values = [1.0/3, 1.0/3, 1.0/3]
    else:
        # Normalize to sum to 1.0
        values = [v / total for v in values]
    
    return values


@st.composite
def valid_profile_config(draw):
    """Generate a valid profile configuration."""
    query_dist = draw(valid_distribution())
    complexity_dist = draw(valid_distribution())
    
    return {
        'name': draw(st.text(min_size=1, max_size=50)),
        'description': draw(st.text(min_size=1, max_size=200)),
        'query_distribution': {
            'visual': query_dist[0],
            'code': query_dist[1],
            'research': query_dist[2]
        },
        'complexity_distribution': {
            'simple': complexity_dist[0],
            'medium': complexity_dist[1],
            'complex': complexity_dist[2]
        },
        'routing_accuracy': draw(st.floats(min_value=0.0, max_value=1.0)),
        'delegation_accuracy': draw(st.floats(min_value=0.0, max_value=1.0)),
        'queries_per_day': draw(st.integers(min_value=1, max_value=10000))
    }


@st.composite
def valid_pricing_config(draw):
    """Generate a valid pricing configuration."""
    return {
        'models': {
            'gemini': {
                'input': draw(st.floats(min_value=0.0, max_value=1.0)),
                'output': draw(st.floats(min_value=0.0, max_value=1.0))
            },
            'coder': {
                'input': draw(st.floats(min_value=0.0, max_value=1.0)),
                'output': draw(st.floats(min_value=0.0, max_value=1.0))
            },
            'grok': {
                'input': draw(st.floats(min_value=0.0, max_value=1.0)),
                'output': draw(st.floats(min_value=0.0, max_value=1.0))
            },
            'classifier': {
                'input': draw(st.floats(min_value=0.0, max_value=1.0)),
                'output': draw(st.floats(min_value=0.0, max_value=1.0))
            }
        }
    }


@st.composite
def valid_token_estimation_config(draw):
    """Generate a valid token estimation configuration."""
    return {
        'system_prompts': {
            'gemini': draw(st.integers(min_value=1, max_value=1000)),
            'coder': draw(st.integers(min_value=1, max_value=1000)),
            'grok': draw(st.integers(min_value=1, max_value=1000)),
            'classifier': draw(st.integers(min_value=1, max_value=1000))
        },
        'input_ranges': {
            'visual': {
                'simple': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'medium': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'complex': [draw(st.integers(min_value=1, max_value=100)), 
                           draw(st.integers(min_value=101, max_value=500))]
            },
            'code': {
                'simple': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'medium': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'complex': [draw(st.integers(min_value=1, max_value=100)), 
                           draw(st.integers(min_value=101, max_value=500))]
            },
            'research': {
                'simple': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'medium': [draw(st.integers(min_value=1, max_value=100)), 
                          draw(st.integers(min_value=101, max_value=500))],
                'complex': [draw(st.integers(min_value=1, max_value=100)), 
                           draw(st.integers(min_value=101, max_value=500))]
            }
        },
        'output_multipliers': {
            'simple': draw(st.floats(min_value=0.1, max_value=2.0)),
            'medium': draw(st.floats(min_value=0.1, max_value=2.0)),
            'complex': draw(st.floats(min_value=0.1, max_value=2.0))
        },
        'wrong_model_penalty': draw(st.floats(min_value=1.0, max_value=3.0))
    }


@st.composite
def valid_simulation_config(draw):
    """Generate a valid simulation configuration."""
    return {
        'runs': draw(st.integers(min_value=1, max_value=1000)),
        'days_to_simulate': draw(st.integers(min_value=1, max_value=365)),
        'sensitivity_analysis': {
            'routing_accuracy_range': draw(st.lists(
                st.floats(min_value=0.0, max_value=1.0), 
                min_size=2, max_size=10
            )),
            'query_volume_range': draw(st.lists(
                st.integers(min_value=1, max_value=10000), 
                min_size=2, max_size=10
            )),
            'complex_query_percentage': draw(st.lists(
                st.floats(min_value=0.0, max_value=1.0), 
                min_size=2, max_size=10
            ))
        }
    }


class TestConfigurationLoadingCompleteness:
    """
    Property 1: Configuration Loading Completeness
    
    For any valid YAML configuration file containing all required fields,
    loading the configuration should successfully parse all fields and make
    them accessible in the resulting configuration object.
    
    **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """

    @given(valid_profile_config())
    def test_profile_loading_completeness(self, profile_config):
        """
        Test that valid profile configurations can be loaded completely.
        
        **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
        **Validates: Requirements 1.1, 1.2**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config structure
            profiles_dir = Path(temp_dir) / "profiles"
            profiles_dir.mkdir()
            
            # Write profile to temporary file
            profile_path = profiles_dir / "test_profile.yaml"
            with open(profile_path, 'w') as f:
                yaml.dump(profile_config, f)
            
            # Mock the config directory to point to our temp directory
            original_get_config_dir = get_config_dir
            try:
                import backend.core.config_loader
                backend.core.config_loader.get_config_dir = lambda: Path(temp_dir)
                
                # Load the profile
                loaded_config = load_profile("test_profile")
                
                # Verify all fields are accessible and match original
                assert loaded_config['name'] == profile_config['name']
                assert loaded_config['description'] == profile_config['description']
                
                # Verify query distribution
                assert 'query_distribution' in loaded_config
                for key in ['visual', 'code', 'research']:
                    assert key in loaded_config['query_distribution']
                    assert loaded_config['query_distribution'][key] == profile_config['query_distribution'][key]
                
                # Verify complexity distribution
                assert 'complexity_distribution' in loaded_config
                for key in ['simple', 'medium', 'complex']:
                    assert key in loaded_config['complexity_distribution']
                    assert loaded_config['complexity_distribution'][key] == profile_config['complexity_distribution'][key]
                
                # Verify accuracy values
                assert loaded_config['routing_accuracy'] == profile_config['routing_accuracy']
                assert loaded_config['delegation_accuracy'] == profile_config['delegation_accuracy']
                
                # Verify queries per day
                assert loaded_config['queries_per_day'] == profile_config['queries_per_day']
                
            finally:
                # Restore original function
                backend.core.config_loader.get_config_dir = original_get_config_dir

    @given(valid_pricing_config())
    def test_pricing_loading_completeness(self, pricing_config):
        """
        Test that valid pricing configurations can be loaded completely.
        
        **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
        **Validates: Requirements 1.1, 1.3**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write pricing to temporary file
            pricing_path = Path(temp_dir) / "pricing.yaml"
            with open(pricing_path, 'w') as f:
                yaml.dump(pricing_config, f)
            
            # Mock the config directory
            original_get_config_dir = get_config_dir
            try:
                import backend.core.config_loader
                backend.core.config_loader.get_config_dir = lambda: Path(temp_dir)
                
                # Load the pricing config
                loaded_config = load_pricing()
                
                # Verify all models and their pricing are accessible
                assert 'models' in loaded_config
                for model_name in ['gemini', 'coder', 'grok', 'classifier']:
                    assert model_name in loaded_config['models']
                    assert 'input' in loaded_config['models'][model_name]
                    assert 'output' in loaded_config['models'][model_name]
                    assert loaded_config['models'][model_name]['input'] == pricing_config['models'][model_name]['input']
                    assert loaded_config['models'][model_name]['output'] == pricing_config['models'][model_name]['output']
                
            finally:
                backend.core.config_loader.get_config_dir = original_get_config_dir

    @given(valid_token_estimation_config())
    def test_token_estimation_loading_completeness(self, token_config):
        """
        Test that valid token estimation configurations can be loaded completely.
        
        **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
        **Validates: Requirements 1.1, 1.4**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write token estimation config to temporary file
            token_path = Path(temp_dir) / "token_estimation.yaml"
            with open(token_path, 'w') as f:
                yaml.dump(token_config, f)
            
            # Mock the config directory
            original_get_config_dir = get_config_dir
            try:
                import backend.core.config_loader
                backend.core.config_loader.get_config_dir = lambda: Path(temp_dir)
                
                # Load the token estimation config
                loaded_config = load_token_estimation()
                
                # Verify system prompts
                assert 'system_prompts' in loaded_config
                for model in ['gemini', 'coder', 'grok', 'classifier']:
                    assert model in loaded_config['system_prompts']
                    assert loaded_config['system_prompts'][model] == token_config['system_prompts'][model]
                
                # Verify input ranges
                assert 'input_ranges' in loaded_config
                for query_type in ['visual', 'code', 'research']:
                    assert query_type in loaded_config['input_ranges']
                    for complexity in ['simple', 'medium', 'complex']:
                        assert complexity in loaded_config['input_ranges'][query_type]
                        assert loaded_config['input_ranges'][query_type][complexity] == token_config['input_ranges'][query_type][complexity]
                
                # Verify output multipliers
                assert 'output_multipliers' in loaded_config
                for complexity in ['simple', 'medium', 'complex']:
                    assert complexity in loaded_config['output_multipliers']
                    assert loaded_config['output_multipliers'][complexity] == token_config['output_multipliers'][complexity]
                
                # Verify wrong model penalty
                assert 'wrong_model_penalty' in loaded_config
                assert loaded_config['wrong_model_penalty'] == token_config['wrong_model_penalty']
                
            finally:
                backend.core.config_loader.get_config_dir = original_get_config_dir

    @given(valid_simulation_config())
    def test_simulation_loading_completeness(self, simulation_config):
        """
        Test that valid simulation configurations can be loaded completely.
        
        **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
        **Validates: Requirements 1.1, 1.5**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write simulation config to temporary file
            simulation_path = Path(temp_dir) / "simulation.yaml"
            with open(simulation_path, 'w') as f:
                yaml.dump(simulation_config, f)
            
            # Mock the config directory
            original_get_config_dir = get_config_dir
            try:
                import backend.core.config_loader
                backend.core.config_loader.get_config_dir = lambda: Path(temp_dir)
                
                # Load the simulation config
                loaded_config = load_simulation()
                
                # Verify basic simulation parameters
                assert loaded_config['runs'] == simulation_config['runs']
                assert loaded_config['days_to_simulate'] == simulation_config['days_to_simulate']
                
                # Verify sensitivity analysis parameters
                assert 'sensitivity_analysis' in loaded_config
                sensitivity = loaded_config['sensitivity_analysis']
                original_sensitivity = simulation_config['sensitivity_analysis']
                
                assert sensitivity['routing_accuracy_range'] == original_sensitivity['routing_accuracy_range']
                assert sensitivity['query_volume_range'] == original_sensitivity['query_volume_range']
                assert sensitivity['complex_query_percentage'] == original_sensitivity['complex_query_percentage']
                
            finally:
                backend.core.config_loader.get_config_dir = original_get_config_dir

    @given(st.text(min_size=1))
    def test_yaml_file_loading_completeness(self, content):
        """
        Test that YAML file loading preserves all data structure.
        
        **Feature: llm-cost-predictor, Property 1: Configuration Loading Completeness**
        **Validates: Requirements 1.1**
        """
        # Create a simple but valid YAML structure
        test_data = {
            'string_field': content,
            'number_field': 42,
            'float_field': 3.14,
            'boolean_field': True,
            'list_field': [1, 2, 3],
            'dict_field': {'nested': 'value'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Load the YAML file
            loaded_data = load_yaml_file(Path(temp_path))
            
            # Verify all fields are accessible and preserved
            assert loaded_data['string_field'] == test_data['string_field']
            assert loaded_data['number_field'] == test_data['number_field']
            assert loaded_data['float_field'] == test_data['float_field']
            assert loaded_data['boolean_field'] == test_data['boolean_field']
            assert loaded_data['list_field'] == test_data['list_field']
            assert loaded_data['dict_field'] == test_data['dict_field']
            assert loaded_data['dict_field']['nested'] == test_data['dict_field']['nested']
            
        finally:
            os.unlink(temp_path)