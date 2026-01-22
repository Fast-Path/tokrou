# Implementation Plan: Dynamic Model Configuration

## Overview

This implementation refactors the Multi-Model LLM Cost Predictor to support dynamic/configurable models. The work is organized into phases: creating new config files, updating the config loader and validator, modifying types and token estimator, and updating tests.

## Tasks

- [-] 1. Create new configuration files
  - [-] 1.1 Create `config/routing.json` with classifier_model and query_type_routes
    - Define classifier_model as "classifier"
    - Map visual→gemini, code→coder, research→grok
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [~] 1.2 Create `config/delegation.json` with delegation rules
    - Define rules for gemini, coder, grok delegation targets
    - _Requirements: 3.1, 3.2_
  
  - [ ] 1.3 Create `config/model_aliases.json` with category mappings
    - Map GEMINI, CODER, GROK, CLASSIFIER to actual model names
    - _Requirements: 5.1, 5.2_

- [~] 2. Update config_validator.py with new validation functions
  - [ ] 2.1 Add `validate_routing_config()` function
    - Validate classifier_model exists in valid_models
    - Validate all query_type_routes targets exist in valid_models
    - Validate query types are valid enum values
    - Return list of all errors (not just first)
    - _Requirements: 2.4, 2.5, 4.3, 6.5_
  
  - [ ] 2.2 Add `validate_delegation_config()` function
    - Validate all source models exist in valid_models
    - Validate all target models exist in valid_models
    - Validate query types are valid enum values
    - Return list of all errors
    - _Requirements: 3.3, 3.4, 3.5, 4.4, 6.5_
  
  - [ ] 2.3 Add `validate_model_aliases_config()` function
    - Validate all category names exist in valid_models
    - Return list of all errors
    - _Requirements: 4.5, 5.5, 6.4, 6.5_
  
  - [ ] 2.4 Modify `validate_pricing_config()` to accept any model names
    - Remove hardcoded model name validation
    - Keep input/output price validation
    - _Requirements: 4.1, 4.2_

- [~] 3. Update config_loader.py with new loading functions
  - [ ] 3.1 Add `load_routing()` function
    - Load from config/routing.json
    - Validate against pricing config
    - Raise ConfigValidationError on invalid config
    - _Requirements: 2.1, 2.4_
  
  - [ ] 3.2 Add `load_delegation()` function
    - Load from config/delegation.json
    - Validate against pricing config
    - Raise ConfigValidationError on invalid config
    - _Requirements: 3.1, 3.3, 3.4_
  
  - [~] 3.3 Add `load_model_aliases()` function
    - Load from config/model_aliases.json
    - Validate category names against pricing config
    - Raise ConfigValidationError on invalid config
    - _Requirements: 5.1, 5.5_
  
  - [ ] 3.4 Add `get_valid_models()` helper function
    - Extract set of model names from pricing config
    - _Requirements: 4.1_
  
  - [ ] 3.5 Add model alias resolution functions
    - `resolve_model_to_category()` - map model name to category
    - `resolve_category_to_models()` - get models for category
    - _Requirements: 5.3_

- [~] 4. Checkpoint - Verify config loading works
  - Ensure all new config files load successfully
  - Ensure validation catches invalid model references
  - Ask the user if questions arise

- [~] 5. Update types.py to remove ModelName enum
  - [ ] 5.1 Remove ModelName enum class
    - Delete the ModelName enum definition
    - _Requirements: 1.1_
  
  - [ ] 5.2 Update __init__.py exports
    - Remove ModelName from exports in backend/core/__init__.py
    - _Requirements: 1.1_

- [~] 6. Update token_estimator.py to use string model names
  - [ ] 6.1 Remove ModelName import and update type hints
    - Change `model: ModelName` to `model: str` in all methods
    - Remove `from .types import ModelName`
    - _Requirements: 1.3, 8.1_
  
  - [ ] 6.2 Update model lookups to use string keys directly
    - Change `model.value` to `model` in dictionary lookups
    - Add error handling for unknown models
    - _Requirements: 8.2, 8.3_

- [~] 7. Update config/token_estimation.json to support dynamic models
  - [ ] 7.1 Verify token_estimation.json works with current model names
    - No structural changes needed, just verify compatibility
    - _Requirements: 8.4_

- [~] 8. Checkpoint - Verify core changes work
  - Ensure TokenEstimator works with string model names
  - Ensure log parser still works (already uses strings)
  - Ask the user if questions arise

- [~] 9. Update unit tests
  - [ ] 9.1 Update tests/unit/test_types.py
    - Remove test_model_name_values test
    - Keep QueryType and Complexity tests
    - _Requirements: 1.1_
  
  - [~] 9.2 Add tests for new config loading functions
    - Test load_routing() with valid config
    - Test load_delegation() with valid config
    - Test load_model_aliases() with valid config
    - Test validation errors for invalid configs
    - _Requirements: 2.1, 3.1, 5.1_
  
  - [ ] 9.3 Add tests for model alias resolution
    - Test resolve_model_to_category()
    - Test resolve_category_to_models()
    - _Requirements: 5.3_

- [~] 10. Update property tests
  - [~] 10.1 Write property test for configuration loading round-trip

    - **Property 1: Configuration Loading Round-Trip**
    - **Validates: Requirements 2.1, 2.2, 2.3, 3.1, 3.2, 5.1, 5.2**
  
  - [ ]* 10.2 Write property test for model validation against pricing
    - **Property 2: Model Validation Against Pricing**
    - **Validates: Requirements 1.4, 2.4, 3.3, 3.4, 4.2, 4.3, 4.4, 4.5, 5.5**
  
  - [ ]* 10.3 Write property test for query type validation
    - **Property 3: Query Type Validation**
    - **Validates: Requirements 2.5, 3.5**
  
  - [ ]* 10.4 Write property test for log parsing with arbitrary strings
    - **Property 4: Log Parsing with Arbitrary Model Strings**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
  
  - [ ]* 10.5 Write property test for token estimation with string models
    - **Property 5: Token Estimation with String Models**
    - **Validates: Requirements 1.3, 8.1, 8.2**
  
  - [ ]* 10.6 Write property test for token estimation error on unknown models
    - **Property 6: Token Estimation Error for Unknown Models**
    - **Validates: Requirements 8.3**
  
  - [ ]* 10.7 Write property test for validation error collection
    - **Property 7: Validation Error Collection**
    - **Validates: Requirements 6.5**
  
  - [ ]* 10.8 Write property test for model-to-category resolution
    - **Property 8: Model-to-Category Resolution**
    - **Validates: Requirements 5.3**

- [ ] 11. Update existing property tests
  - [ ] 11.1 Update test_config_properties.py pricing config generator
    - Remove hardcoded model name requirement
    - Generate arbitrary model names
    - _Requirements: 4.1, 4.2_
  
  - [ ] 11.2 Update test_config_properties.py token estimation config generator
    - Remove hardcoded model name requirement
    - Generate arbitrary model names matching pricing config
    - _Requirements: 8.4_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Run full test suite
  - Verify no regressions
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- QueryType and Complexity enums are intentionally kept as they represent domain concepts
- The log parser already uses strings, so minimal changes needed there
- Property tests validate universal correctness properties across random inputs
