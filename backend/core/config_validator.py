"""Configuration validation utilities."""

from typing import Dict, List, Any


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_distribution(
    distribution: Dict[str, float],
    name: str,
    tolerance: float = 0.001
) -> List[str]:
    """
    Validate that distribution values sum to 1.0.
    
    Args:
        distribution: Dictionary of distribution values
        name: Name of the distribution for error messages
        tolerance: Acceptable deviation from 1.0
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check all values are floats
    for key, value in distribution.items():
        if not isinstance(value, (int, float)):
            errors.append(f"{name}.{key} must be a number, got {type(value).__name__}")
            continue
        
        # Check values are non-negative
        if value < 0:
            errors.append(f"{name}.{key} must be non-negative, got {value}")
    
    # Check sum equals 1.0 within tolerance
    total = sum(distribution.values())
    if abs(total - 1.0) > tolerance:
        errors.append(
            f"{name} values must sum to 1.0 (±{tolerance}), got {total:.6f}"
        )
    
    return errors


def validate_accuracy(
    value: float,
    name: str
) -> List[str]:
    """
    Validate that accuracy value is between 0.0 and 1.0 inclusive.
    
    Args:
        value: Accuracy value to validate
        name: Name of the field for error messages
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not isinstance(value, (int, float)):
        errors.append(f"{name} must be a number, got {type(value).__name__}")
        return errors
    
    if value < 0.0 or value > 1.0:
        errors.append(f"{name} must be between 0.0 and 1.0, got {value}")
    
    return errors


def validate_positive_integer(
    value: int,
    name: str
) -> List[str]:
    """
    Validate that value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the field for error messages
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not isinstance(value, int):
        errors.append(f"{name} must be an integer, got {type(value).__name__}")
        return errors
    
    if value <= 0:
        errors.append(f"{name} must be positive, got {value}")
    
    return errors


def validate_profile_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a complete usage profile configuration.
    
    Args:
        config: Profile configuration dictionary
    
    Returns:
        List of validation error messages (empty if valid)
    
    Validates:
        - Required fields present
        - Distributions sum to 1.0
        - Accuracy values in valid range
        - Positive query volume
    """
    errors = []
    
    # Check required fields
    required_fields = [
        'name',
        'description',
        'query_distribution',
        'complexity_distribution',
        'routing_accuracy',
        'delegation_accuracy',
        'queries_per_day'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # If missing required fields, return early
    if errors:
        return errors
    
    # Validate query distribution
    if isinstance(config['query_distribution'], dict):
        required_query_types = {'visual', 'code', 'research'}
        actual_query_types = set(config['query_distribution'].keys())
        
        if actual_query_types != required_query_types:
            errors.append(
                f"query_distribution must contain exactly {required_query_types}, "
                f"got {actual_query_types}"
            )
        else:
            errors.extend(
                validate_distribution(
                    config['query_distribution'],
                    'query_distribution'
                )
            )
    else:
        errors.append("query_distribution must be a dictionary")
    
    # Validate complexity distribution
    if isinstance(config['complexity_distribution'], dict):
        required_complexities = {'simple', 'medium', 'complex'}
        actual_complexities = set(config['complexity_distribution'].keys())
        
        if actual_complexities != required_complexities:
            errors.append(
                f"complexity_distribution must contain exactly {required_complexities}, "
                f"got {actual_complexities}"
            )
        else:
            errors.extend(
                validate_distribution(
                    config['complexity_distribution'],
                    'complexity_distribution'
                )
            )
    else:
        errors.append("complexity_distribution must be a dictionary")
    
    # Validate routing accuracy
    errors.extend(
        validate_accuracy(
            config['routing_accuracy'],
            'routing_accuracy'
        )
    )
    
    # Validate delegation accuracy
    errors.extend(
        validate_accuracy(
            config['delegation_accuracy'],
            'delegation_accuracy'
        )
    )
    
    # Validate queries per day
    errors.extend(
        validate_positive_integer(
            config['queries_per_day'],
            'queries_per_day'
        )
    )
    
    return errors


def validate_pricing_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate pricing configuration.
    
    Args:
        config: Pricing configuration dictionary
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if 'models' not in config:
        errors.append("Missing required field: models")
        return errors
    
    required_models = {'gemini', 'coder', 'grok', 'classifier'}
    actual_models = set(config['models'].keys())
    
    if actual_models != required_models:
        errors.append(
            f"models must contain exactly {required_models}, got {actual_models}"
        )
        return errors
    
    # Validate each model has input and output pricing
    for model_name, pricing in config['models'].items():
        if not isinstance(pricing, dict):
            errors.append(f"models.{model_name} must be a dictionary")
            continue
        
        if 'input' not in pricing:
            errors.append(f"models.{model_name} missing 'input' price")
        elif not isinstance(pricing['input'], (int, float)) or pricing['input'] < 0:
            errors.append(f"models.{model_name}.input must be a non-negative number")
        
        if 'output' not in pricing:
            errors.append(f"models.{model_name} missing 'output' price")
        elif not isinstance(pricing['output'], (int, float)) or pricing['output'] < 0:
            errors.append(f"models.{model_name}.output must be a non-negative number")
    
    return errors


def validate_token_estimation_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate token estimation configuration.
    
    Args:
        config: Token estimation configuration dictionary
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = [
        'system_prompts',
        'input_ranges',
        'output_multipliers',
        'wrong_model_penalty'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Validate system prompts
    required_models = {'gemini', 'coder', 'grok', 'classifier'}
    if set(config['system_prompts'].keys()) != required_models:
        errors.append(
            f"system_prompts must contain {required_models}"
        )
    
    # Validate input ranges
    required_query_types = {'visual', 'code', 'research'}
    required_complexities = {'simple', 'medium', 'complex'}
    
    if set(config['input_ranges'].keys()) != required_query_types:
        errors.append(
            f"input_ranges must contain {required_query_types}"
        )
    else:
        for query_type, ranges in config['input_ranges'].items():
            if set(ranges.keys()) != required_complexities:
                errors.append(
                    f"input_ranges.{query_type} must contain {required_complexities}"
                )
    
    # Validate output multipliers
    if set(config['output_multipliers'].keys()) != required_complexities:
        errors.append(
            f"output_multipliers must contain {required_complexities}"
        )
    
    # Validate wrong model penalty
    if not isinstance(config['wrong_model_penalty'], (int, float)):
        errors.append("wrong_model_penalty must be a number")
    elif config['wrong_model_penalty'] < 1.0:
        errors.append("wrong_model_penalty must be >= 1.0")
    
    return errors
