# Requirements Document

## Introduction

This feature refactors the Multi-Model LLM Cost Predictor to support dynamic/configurable models instead of hardcoded enums. Currently, the `ModelName` enum in `backend/core/types.py` hardcodes models (gemini, coder, grok, classifier), requiring Python code changes to add, remove, or rename models. This refactoring makes the system config-driven so models can be changed via JSON files only.

## Glossary

- **Config_Loader**: The module responsible for loading and parsing JSON configuration files
- **Config_Validator**: The module responsible for validating configuration data against defined rules
- **Model_Category**: A logical grouping of models (e.g., GEMINI, CODER, GROK) used for routing decisions
- **Model_Alias**: An actual model name (e.g., "grok-4-1-fast") that maps to a Model_Category
- **Pricing_Config**: The configuration file (`config/pricing.json`) that defines available models and their token costs
- **Routing_Config**: The configuration file (`config/routing.json`) that defines how queries are routed to models
- **Delegation_Config**: The configuration file (`config/delegation.json`) that defines delegation rules between models
- **Alias_Config**: The configuration file (`config/model_aliases.json`) that maps actual model names to categories
- **Query_Type**: The classification of a query (visual, code, research) - remains as an enum
- **Complexity**: The complexity level of a query (simple, medium, complex) - remains as an enum

## Requirements

### Requirement 1: Remove Hardcoded ModelName Enum

**User Story:** As a developer, I want model names to be strings loaded from configuration, so that I can add or modify models without changing Python code.

#### Acceptance Criteria

1. THE Config_Loader SHALL load model names as strings from configuration files instead of using enum values
2. THE CostBreakdown dataclass SHALL use string type for model-related fields instead of ModelName enum
3. THE Token_Estimator SHALL accept string model names as parameters instead of ModelName enum values
4. WHEN a model name is used in the system THEN the Config_Validator SHALL validate it against models defined in Pricing_Config

### Requirement 2: Create Routing Configuration

**User Story:** As a system administrator, I want to configure query routing rules in a JSON file, so that I can change routing behavior without code changes.

#### Acceptance Criteria

1. THE Config_Loader SHALL load routing configuration from `config/routing.json`
2. THE Routing_Config SHALL define a classifier_model field specifying which model handles classification
3. THE Routing_Config SHALL define query_type_routes mapping each Query_Type to a target model
4. WHEN Routing_Config is loaded THEN the Config_Validator SHALL verify all referenced models exist in Pricing_Config
5. WHEN Routing_Config references a Query_Type THEN the Config_Validator SHALL verify it matches valid Query_Type enum values (visual, code, research)

### Requirement 3: Create Delegation Configuration

**User Story:** As a system administrator, I want to configure delegation rules in a JSON file, so that I can modify how models delegate to each other without code changes.

#### Acceptance Criteria

1. THE Config_Loader SHALL load delegation configuration from `config/delegation.json`
2. THE Delegation_Config SHALL define rules mapping source models to delegation targets by Query_Type
3. WHEN Delegation_Config is loaded THEN the Config_Validator SHALL verify all source models exist in Pricing_Config
4. WHEN Delegation_Config is loaded THEN the Config_Validator SHALL verify all target models exist in Pricing_Config
5. WHEN Delegation_Config references a Query_Type THEN the Config_Validator SHALL verify it matches valid Query_Type enum values

### Requirement 4: Pricing Config as Source of Truth

**User Story:** As a developer, I want pricing.json to be the authoritative source for valid models, so that model validation is centralized and consistent.

#### Acceptance Criteria

1. THE Pricing_Config SHALL define all valid models in the system through its models object
2. THE Config_Validator SHALL accept any model defined in Pricing_Config as valid (not restricted to a fixed set)
3. WHEN validating Routing_Config THEN the Config_Validator SHALL reject any model not present in Pricing_Config
4. WHEN validating Delegation_Config THEN the Config_Validator SHALL reject any model not present in Pricing_Config
5. WHEN validating Alias_Config THEN the Config_Validator SHALL reject any category model not present in Pricing_Config

### Requirement 5: Model Alias/Category Mapping

**User Story:** As a system administrator, I want to map actual model names to categories, so that log data using specific model names can be correctly associated with routing categories.

#### Acceptance Criteria

1. THE Config_Loader SHALL load model alias configuration from `config/model_aliases.json`
2. THE Alias_Config SHALL define categories mapping category names to lists of actual model names
3. WHEN a log entry contains a model_used value THEN the system SHALL be able to resolve it to a category via Alias_Config
4. WHEN a log entry contains a routed_to value THEN the system SHALL recognize it as a category name
5. WHEN Alias_Config is loaded THEN the Config_Validator SHALL verify all category names exist as models in Pricing_Config

### Requirement 6: Cross-Configuration Validation

**User Story:** As a developer, I want the system to validate that all configuration files are consistent with each other, so that configuration errors are caught at load time.

#### Acceptance Criteria

1. WHEN all configuration files are loaded THEN the Config_Validator SHALL verify cross-file consistency
2. IF Routing_Config references a model not in Pricing_Config THEN the Config_Validator SHALL return a descriptive error
3. IF Delegation_Config references a model not in Pricing_Config THEN the Config_Validator SHALL return a descriptive error
4. IF Alias_Config references a category not in Pricing_Config THEN the Config_Validator SHALL return a descriptive error
5. THE Config_Validator SHALL collect all validation errors and return them together rather than failing on the first error

### Requirement 7: Backward Compatibility for Log Parsing

**User Story:** As a data analyst, I want the log parser to continue working with existing log formats, so that historical data remains usable.

#### Acceptance Criteria

1. THE Log_Parser SHALL continue to accept log entries with string model names in model_used field
2. THE Log_Parser SHALL continue to accept log entries with category names in routed_to field
3. THE Log_Parser SHALL continue to accept log entries with model name lists in delegation_chain field
4. WHEN parsing log entries THEN the Log_Parser SHALL not require model names to match any enum

### Requirement 8: Token Estimation with Dynamic Models

**User Story:** As a developer, I want token estimation to work with dynamically configured models, so that cost calculations remain accurate.

#### Acceptance Criteria

1. THE Token_Estimator SHALL accept string model names instead of ModelName enum values
2. WHEN estimating tokens for a model THEN the Token_Estimator SHALL look up system prompt size from configuration
3. IF a model is not found in token estimation config THEN the Token_Estimator SHALL raise a descriptive error
4. THE token_estimation.json config SHALL support arbitrary model names (not restricted to a fixed set)
