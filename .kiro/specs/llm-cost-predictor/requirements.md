# Requirements Document

## Introduction

This document specifies the requirements for a Multi-Model LLM Cost Predictor system. The system predicts operational costs for a multi-model LLM orchestration architecture (Gemini + Qwen/Coder + Grok) before production deployment using Monte Carlo simulation. The system provides both CLI and web interfaces for cost analysis, sensitivity testing, and scenario comparison.

## Glossary

- **System**: The Multi-Model LLM Cost Predictor application
- **CLI**: Command-line interface for running simulations and generating reports
- **Web_Interface**: React-based frontend with Flask backend for interactive exploration
- **Cost_Model**: Core calculation logic for predicting query costs
- **Monte_Carlo_Simulation**: Statistical method using repeated random sampling to predict cost distributions
- **Profile**: Configuration defining query distributions, complexity, routing accuracy, and volume
- **Query_Type**: Classification of queries (VISUAL, CODE, RESEARCH)
- **Complexity**: Query complexity level (SIMPLE, MEDIUM, COMPLEX)
- **Lead_Model**: Primary model handling a query (Gemini, Coder, or Grok)
- **Classifier**: Lightweight routing model that directs queries to appropriate Lead_Model
- **Delegation**: When a Lead_Model invokes another model to complete a task
- **Context_Reinjection**: Process where Lead_Model reprocesses original query plus tool results
- **Routing_Accuracy**: Percentage of queries correctly routed to optimal Lead_Model
- **Token_Estimator**: Component predicting token usage for queries
- **Sensitivity_Analysis**: Testing how cost varies with different parameter values

## Requirements

### Requirement 1: Configuration Management

**User Story:** As a system administrator, I want all system parameters externally configurable via YAML files, so that I can adjust pricing, usage patterns, and simulation parameters without code changes.

#### Acceptance Criteria

1. WHEN the System loads configuration files, THE System SHALL parse YAML files from the config directory
2. THE System SHALL load usage profiles from config/profiles/ directory containing query distributions, complexity distributions, routing accuracy, delegation accuracy, and queries per day
3. THE System SHALL load model pricing from config/pricing.yaml containing per-token input and output costs for all models
4. THE System SHALL load token estimation rules from config/token_estimation.yaml containing system prompt sizes, input token ranges, output multipliers, and wrong model penalties
5. THE System SHALL load simulation parameters from config/simulation.yaml containing number of runs, days to simulate, and sensitivity analysis ranges
6. WHEN configuration values are invalid, THEN THE System SHALL report descriptive validation errors
7. THE System SHALL validate that distribution values sum to 1.0 with tolerance of 0.001
8. THE System SHALL validate that accuracy values are between 0.0 and 1.0 inclusive

### Requirement 2: Token Estimation

**User Story:** As a cost analyst, I want accurate token usage predictions for different query types and complexities, so that cost simulations reflect realistic API usage.

#### Acceptance Criteria

1. WHEN estimating tokens for a query, THE Token_Estimator SHALL include system prompt tokens for the target model
2. WHEN estimating input tokens, THE Token_Estimator SHALL sample from the configured range for the query type and complexity
3. WHEN estimating output tokens, THE Token_Estimator SHALL multiply input tokens by the complexity-specific output multiplier
4. WHEN a query is routed to the wrong model, THE Token_Estimator SHALL apply the wrong model penalty multiplier to token estimates
5. THE Token_Estimator SHALL estimate classifier tokens for routing decisions
6. WHEN calculating delegation tokens, THE Token_Estimator SHALL estimate tokens for both tool execution and context reinjection
7. THE Token_Estimator SHALL support deterministic token estimation when provided with a random seed

### Requirement 3: Single Query Cost Calculation

**User Story:** As a developer, I want to calculate the cost of individual queries, so that I can understand cost breakdown by component.

#### Acceptance Criteria

1. WHEN calculating query cost, THE Cost_Model SHALL compute routing cost as classifier token usage multiplied by classifier pricing
2. WHEN selecting the lead model, THE Cost_Model SHALL map VISUAL queries to Gemini, CODE queries to Coder, and RESEARCH queries to Grok
3. WHEN simulating routing, THE Cost_Model SHALL route to the correct lead model with probability equal to routing accuracy
4. WHEN a query is misrouted, THE Cost_Model SHALL add correction cost by routing to the correct model after initial attempt
5. WHEN calculating lead model cost, THE Cost_Model SHALL multiply estimated tokens by model-specific pricing for input and output tokens
6. WHEN a MEDIUM complexity query requires delegation, THE Cost_Model SHALL add delegation cost with 80% probability
7. WHEN a COMPLEX complexity query requires delegation, THE Cost_Model SHALL add delegation cost with 100% probability
8. WHEN calculating delegation cost, THE Cost_Model SHALL include both tool execution cost and context reinjection cost
9. WHEN a COMPLEX query has a second delegation, THE Cost_Model SHALL add second delegation cost with 30% probability at 60% of first delegation cost
10. THE Cost_Model SHALL return a cost breakdown containing routing cost, lead cost, delegation cost, total cost, query type, complexity, routing correctness, and number of delegations

### Requirement 4: Delegation Rules

**User Story:** As a system architect, I want delegation rules enforced according to model capabilities, so that simulations reflect actual system constraints.

#### Acceptance Criteria

1. WHEN Gemini delegates for code tasks, THE Cost_Model SHALL route delegation to Coder
2. WHEN Gemini delegates for research tasks, THE Cost_Model SHALL route delegation to Grok
3. WHEN Coder delegates, THE Cost_Model SHALL route delegation to Grok
4. WHEN Grok delegates, THE Cost_Model SHALL route delegation to Coder
5. THE Cost_Model SHALL prevent invalid delegation paths not matching these rules

### Requirement 5: Monte Carlo Simulation

**User Story:** As a cost analyst, I want to run Monte Carlo simulations with configurable parameters, so that I can predict cost distributions over time.

#### Acceptance Criteria

1. WHEN running a simulation, THE System SHALL execute the configured number of simulation runs
2. WHEN simulating each run, THE System SHALL simulate the configured number of days
3. WHEN simulating each day, THE System SHALL generate costs for the configured queries per day
4. WHEN generating each query, THE System SHALL sample query type from the profile's query distribution
5. WHEN generating each query, THE System SHALL sample complexity from the profile's complexity distribution
6. WHEN a simulation completes, THE System SHALL return mean daily cost, median daily cost, P95 daily cost, P99 daily cost, monthly estimate, and standard deviation
7. WHEN a simulation completes, THE System SHALL return cost breakdown percentages for routing, lead, and delegation costs
8. WHEN a simulation completes, THE System SHALL return average delegations per query and realized routing accuracy
9. THE System SHALL support reproducible simulations when provided with a random seed

### Requirement 6: Sensitivity Analysis

**User Story:** As a decision maker, I want to analyze how costs change with different parameters, so that I can identify cost drivers and optimization opportunities.

#### Acceptance Criteria

1. WHEN running sensitivity analysis, THE System SHALL vary routing accuracy across configured values
2. WHEN running sensitivity analysis, THE System SHALL vary query volume across configured values
3. WHEN running sensitivity analysis, THE System SHALL vary complex query percentage across configured values
4. WHEN sensitivity analysis completes, THE System SHALL return baseline cost estimate
5. WHEN sensitivity analysis completes, THE System SHALL return cost estimates for each parameter variation
6. WHEN sensitivity analysis completes, THE System SHALL return percentage delta from baseline for each scenario
7. THE System SHALL support custom parameter ranges for sensitivity analysis

### Requirement 7: CLI Interface

**User Story:** As a developer, I want a command-line interface for running simulations, so that I can integrate cost prediction into scripts and automation.

#### Acceptance Criteria

1. WHEN running the simulate command with a profile name, THE CLI SHALL load the named profile from config/profiles/
2. WHEN running the simulate command with a profile file path, THE CLI SHALL load the profile from the specified file
3. WHEN running the simulate command with parameter overrides, THE CLI SHALL apply overrides to the loaded profile
4. WHEN a simulation completes, THE CLI SHALL output a formatted text report with monthly estimate, daily averages, P95, P99, cost breakdown, and metrics
5. WHEN running the sensitivity command, THE CLI SHALL perform sensitivity analysis and output results in JSON format
6. WHEN running the compare command with multiple profile names, THE CLI SHALL run simulations for each profile and output comparison results
7. WHEN running the report command, THE CLI SHALL generate a full report in the specified format (text, JSON, or markdown)
8. WHEN configuration is invalid, THE CLI SHALL display descriptive error messages
9. THE CLI SHALL support --output flag to write results to a file

### Requirement 8: Flask API

**User Story:** As a frontend developer, I want REST API endpoints for simulations and configuration, so that I can build interactive web interfaces.

#### Acceptance Criteria

1. WHEN receiving POST /api/simulate with profile and simulation parameters, THE API SHALL run a simulation and return projection, breakdown, and metrics
2. WHEN receiving POST /api/sensitivity with baseline profile and parameters to vary, THE API SHALL run sensitivity analysis and return baseline and scenario results
3. WHEN receiving GET /api/profiles, THE API SHALL return a list of available profile names from config/profiles/
4. WHEN receiving GET /api/config/pricing, THE API SHALL return current pricing configuration
5. THE API SHALL validate request payloads and return 400 status with error details for invalid requests
6. THE API SHALL return responses in JSON format
7. THE API SHALL complete simulation requests in under 2 seconds for typical parameters

### Requirement 9: Web Interface - Configuration Editor

**User Story:** As a cost analyst, I want to interactively edit usage profiles, so that I can explore different scenarios without editing YAML files.

#### Acceptance Criteria

1. WHEN the ConfigEditor component loads, THE Web_Interface SHALL display sliders for query distribution (visual, code, research)
2. WHEN the ConfigEditor component loads, THE Web_Interface SHALL display sliders for complexity distribution (simple, medium, complex)
3. WHEN adjusting distribution sliders, THE Web_Interface SHALL ensure values sum to 1.0 and display validation feedback
4. WHEN the ConfigEditor component loads, THE Web_Interface SHALL display input for routing accuracy with range 0.0 to 1.0
5. WHEN the ConfigEditor component loads, THE Web_Interface SHALL display input for queries per day
6. WHEN loading a profile, THE Web_Interface SHALL populate all fields with profile values
7. WHEN configuration is invalid, THE Web_Interface SHALL disable the run simulation button and display validation errors

### Requirement 10: Web Interface - Simulation Runner

**User Story:** As a cost analyst, I want to trigger simulations from the web interface, so that I can see results without using the CLI.

#### Acceptance Criteria

1. WHEN the SimulationRunner component loads, THE Web_Interface SHALL display a profile selection dropdown
2. WHEN the SimulationRunner component loads, THE Web_Interface SHALL display a run simulation button
3. WHEN clicking run simulation, THE Web_Interface SHALL send a request to POST /api/simulate
4. WHEN a simulation is running, THE Web_Interface SHALL display a progress indicator
5. WHEN a simulation completes, THE Web_Interface SHALL display results in the CostDashboard component
6. THE Web_Interface SHALL remain responsive during simulation execution

### Requirement 11: Web Interface - Cost Dashboard

**User Story:** As a cost analyst, I want to visualize simulation results, so that I can quickly understand cost projections and breakdowns.

#### Acceptance Criteria

1. WHEN simulation results are available, THE Web_Interface SHALL display monthly estimate, P95 daily cost, and P99 daily cost in metric cards
2. WHEN simulation results are available, THE Web_Interface SHALL display a pie chart showing cost breakdown by routing, lead, and delegation
3. WHEN simulation results are available, THE Web_Interface SHALL display a histogram showing distribution of daily costs
4. THE Web_Interface SHALL format monetary values with 2 decimal places
5. THE Web_Interface SHALL format percentages with 1 decimal place
6. WHEN comparing multiple scenarios, THE Web_Interface SHALL display a comparison table

### Requirement 12: Web Interface - Sensitivity Analysis

**User Story:** As a decision maker, I want to visualize parameter sensitivity, so that I can identify which factors most impact costs.

#### Acceptance Criteria

1. WHEN sensitivity analysis results are available, THE Web_Interface SHALL display a line chart showing cost versus parameter values
2. WHEN sensitivity analysis results are available, THE Web_Interface SHALL display a tornado chart showing impact magnitude sorted by parameter
3. WHEN hovering over chart elements, THE Web_Interface SHALL display exact values
4. THE Web_Interface SHALL support triggering sensitivity analysis from the SimulationRunner component

### Requirement 13: Validation and Calibration

**User Story:** As a system administrator, I want to validate token estimates against real API usage, so that I can ensure simulation accuracy.

#### Acceptance Criteria

1. THE System SHALL include a validation script at tests/validate_estimates.py
2. WHEN running validation, THE validation script SHALL load hand-crafted test queries spanning all query types and complexities
3. WHEN running validation, THE validation script SHALL execute queries through real APIs (Gemini, Cerebras, xAI)
4. WHEN running validation, THE validation script SHALL compare actual token usage to estimated token usage
5. WHEN running validation, THE validation script SHALL calculate mean absolute percentage error (MAPE)
6. WHEN MAPE exceeds 30%, THE validation script SHALL fail with an error message
7. WHEN validation completes, THE validation script SHALL report MAPE and per-query errors

### Requirement 14: Performance Requirements

**User Story:** As a user, I want fast simulation execution, so that I can iterate quickly on scenarios.

#### Acceptance Criteria

1. WHEN running a simulation with 100 runs and 30 days, THE System SHALL complete in under 10 seconds
2. WHEN the API receives a simulation request, THE API SHALL respond in under 2 seconds for typical parameters
3. THE Web_Interface SHALL remain responsive during simulation execution

### Requirement 15: Data Export

**User Story:** As a cost analyst, I want to export simulation results, so that I can share findings with stakeholders.

#### Acceptance Criteria

1. WHEN using the CLI report command, THE System SHALL support text format output
2. WHEN using the CLI report command, THE System SHALL support JSON format output
3. WHEN using the CLI report command, THE System SHALL support markdown format output
4. WHEN using the Web_Interface, THE System SHALL support exporting results as JSON
5. WHEN using the Web_Interface, THE System SHALL support exporting results as CSV

### Requirement 16: Documentation

**User Story:** As a new user, I want comprehensive documentation, so that I can understand the system and use it effectively.

#### Acceptance Criteria

1. THE System SHALL include a README with installation instructions
2. THE System SHALL include a README with usage examples for CLI and web interface
3. THE System SHALL include docs/cost_model.md explaining calculation logic
4. THE System SHALL include docs/assumptions.md documenting all assumptions and sources
5. THE System SHALL include docs/calibration.md with instructions for updating token estimates
6. THE System SHALL include inline code comments explaining cost model logic
7. THE System SHALL include comments in configuration files explaining each parameter

### Requirement 17: Configuration Templates

**User Story:** As a new user, I want pre-configured usage profiles, so that I can start with reasonable defaults.

#### Acceptance Criteria

1. THE System SHALL include a conservative profile with pessimistic assumptions
2. THE System SHALL include a baseline profile with expected usage patterns
3. THE System SHALL include an optimistic profile with best-case assumptions
4. THE System SHALL include current pricing for Gemini, Coder, Grok, and Classifier models
5. THE System SHALL include token estimation rules based on measured prompt sizes

### Requirement 18: Profile Management

**User Story:** As a cost analyst, I want to create, save, and delete custom usage profiles, so that I can manage multiple scenarios.

#### Acceptance Criteria

1. WHEN creating a new profile via CLI, THE System SHALL save the profile to config/profiles/ directory with the specified name
2. WHEN creating a new profile via API, THE System SHALL validate the profile data and save it to config/profiles/ directory
3. WHEN deleting a profile via CLI, THE System SHALL remove the profile file from config/profiles/ directory
4. WHEN deleting a profile via API, THE System SHALL remove the profile file from config/profiles/ directory
5. THE System SHALL prevent deletion of built-in profiles (conservative, baseline, optimistic)
6. WHEN saving a profile with an existing name, THE System SHALL prompt for confirmation before overwriting
7. THE System SHALL validate profile data before saving (distributions sum to 1.0, valid ranges)
8. WHEN the Web_Interface saves a profile, THE System SHALL persist it to config/profiles/ directory
9. WHEN the Web_Interface deletes a profile, THE System SHALL remove it from config/profiles/ directory and refresh the profile list

