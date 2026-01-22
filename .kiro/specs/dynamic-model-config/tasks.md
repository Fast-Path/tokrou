# Implementation Plan: Dynamic Model Configuration

## Overview

This implementation refactors the Multi-Model LLM Cost Predictor to support dynamic/configurable models. The work is organized into phases: creating new config files, updating the config loader and validator, modifying types and token estimator, and updating tests.

## Tasks

- [~] 1. Create new configuration files
  - [ ] 1.1 Create `config/routing.json` with classifier_model and query_type_routes
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

- [ ] 3. Update config_loader.py with new loading functions
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
  
  - [ ] 3.3 Add `load_model_aliases()` function
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

- [ ] 4. Checkpoint - Verify config loading works
  - Ensure all new config files load successfully
  - Ensure validation catches invalid model references
  - Ask the user if questions arise

- [ ] 5. Update types.py to remove ModelName enum
  - [ ] 5.1 Remove ModelName enum class
    - Delete the ModelName enum definition
    - _Requirements: 1.1_
  
  - [ ] 5.2 Update __init__.py exports
    - Remove ModelName from exports in backend/core/__init__.py
    - _Requirements: 1.1_
