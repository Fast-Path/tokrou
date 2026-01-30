# Requirements Document

## Introduction

The LLM Cost Predictor is a CLI tool for predicting operational costs of multi-model LLM architectures using Monte Carlo simulation. It models a query routing system where a classifier routes queries to specialized lead models (Gemini for visual, Coder for code, Grok for research), which can delegate to other models for complex queries. The tool generates cost estimates, comparisons, and sensitivity analyses with publication-quality visualizations.

## Glossary

- **Simulator**: The Monte Carlo simulation engine that generates cost predictions by running many randomized trials
- **Cost_Calculator**: Component that computes costs based on token usage and model pricing
- **Token_Estimator**: Component that estimates token counts based on query type and complexity
- **Profile**: A JSON configuration defining usage patterns (query mix, complexity distribution, routing accuracy, daily volume)
- **Classifier**: The routing model that directs queries to appropriate lead models
- **Lead_Model**: Primary models (Gemini, Coder, Grok) that handle routed queries
- **Delegation**: When a lead model passes work to another model for complex queries
- **Routing_Accuracy**: Probability that the classifier routes to the correct model
- **Query_Type**: Category of query (visual, code, research)
- **Complexity**: Query difficulty level (simple, moderate, complex)

## Requirements

### Requirement 1: Monte Carlo Simulation Engine

**User Story:** As a cost analyst, I want to run Monte Carlo simulations on usage profiles, so that I can get statistically robust cost predictions with confidence intervals.

#### Acceptance Criteria

1. WHEN a user runs the simulate command with a profile, THE Simulator SHALL execute the specified number of simulation iterations (default 10,000)
2. WHEN simulating a query, THE Simulator SHALL randomly sample query type based on the profile's query mix distribution
3. WHEN simulating a query, THE Simulator SHALL randomly sample complexity based on the profile's complexity distribution
4. WHEN simulating routing, THE Simulator SHALL determine correct/incorrect routing based on the profile's routing accuracy
5. WHEN routing is incorrect, THE Cost_Calculator SHALL apply a cost penalty by using a mismatched model's pricing
6. WHEN a query is complex, THE Simulator SHALL determine delegation based on routing configuration probabilities
7. WHEN delegation occurs, THE Cost_Calculator SHALL include both lead model and delegation model costs
8. THE Simulator SHALL aggregate results across all iterations to produce mean, P50, P95, and P99 cost statistics

### Requirement 2: Token Estimation

**User Story:** As a cost analyst, I want token counts estimated based on query characteristics, so that I can model realistic cost scenarios without actual API calls.

#### Acceptance Criteria

1. WHEN estimating tokens for a query, THE Token_Estimator SHALL look up base token counts from token_estimation.json based on query type
2. WHEN estimating tokens, THE Token_Estimator SHALL apply complexity multipliers to base token counts
3. WHEN delegation occurs, THE Token_Estimator SHALL estimate additional tokens for the delegation chain
4. THE Token_Estimator SHALL produce separate input and output token estimates for cost calculation

### Requirement 3: Cost Calculation

**User Story:** As a cost analyst, I want costs calculated using actual model pricing, so that I can get accurate cost predictions.

#### Acceptance Criteria

1. WHEN calculating cost, THE Cost_Calculator SHALL load per-token pricing from pricing.json for each model
2. WHEN calculating query cost, THE Cost_Calculator SHALL compute: (input_tokens × input_price) + (output_tokens × output_price)
3. WHEN routing is incorrect, THE Cost_Calculator SHALL use the incorrectly routed model's pricing
4. WHEN delegation occurs, THE Cost_Calculator SHALL sum costs across all models in the delegation chain
5. THE Cost_Calculator SHALL support different pricing for classifier, lead models, and delegation models

### Requirement 4: CLI Simulate Command

**User Story:** As a user, I want to run simulations from the command line, so that I can quickly get cost predictions for different scenarios.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor simulate <profile>`, THE CLI SHALL load the specified profile from config/profiles/
2. WHEN simulation completes, THE CLI SHALL display summary statistics (mean, P50, P95, P99) to the console
3. WHEN the --iterations flag is provided, THE Simulator SHALL use the specified iteration count
4. WHEN the --output-dir flag is provided, THE CLI SHALL save plots to the specified directory
5. WHEN the --no-plot flag is provided, THE CLI SHALL skip plot generation
6. IF the specified profile does not exist, THEN THE CLI SHALL display an error message and exit with non-zero status

### Requirement 5: CLI Compare Command

**User Story:** As a user, I want to compare multiple profiles side-by-side, so that I can evaluate different usage scenarios.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor compare <profile1> <profile2> ...`, THE CLI SHALL run simulations for each profile
2. WHEN comparison completes, THE CLI SHALL display a table comparing statistics across all profiles
3. WHEN comparison completes, THE CLI SHALL generate a comparison plot showing cost distributions for all profiles
4. IF any specified profile does not exist, THEN THE CLI SHALL display an error message listing missing profiles

### Requirement 6: CLI Sensitivity Command

**User Story:** As a user, I want to run sensitivity analysis, so that I can understand how parameter changes affect costs.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor sensitivity <profile> --parameter <param>`, THE CLI SHALL vary the specified parameter across a range
2. WHEN running sensitivity analysis, THE Simulator SHALL run simulations at each parameter value
3. WHEN sensitivity analysis completes, THE CLI SHALL display a table showing cost statistics at each parameter value
4. WHEN sensitivity analysis completes, THE CLI SHALL generate a sensitivity chart showing cost vs parameter value
5. THE CLI SHALL support sensitivity analysis for: routing_accuracy, daily_volume, complexity_distribution, delegation_probability

### Requirement 7: Configuration Management

**User Story:** As a user, I want all parameters externalized in JSON config files, so that I can easily modify scenarios without code changes.

#### Acceptance Criteria

1. THE CLI SHALL load model pricing from config/pricing.json
2. THE CLI SHALL load usage profiles from config/profiles/*.json
3. THE CLI SHALL load token estimation rules from config/token_estimation.json
4. THE CLI SHALL load routing configuration from config/routing.json
5. IF a required config file is missing, THEN THE CLI SHALL display a descriptive error message
6. IF a config file contains invalid JSON, THEN THE CLI SHALL display a parse error with file location

### Requirement 8: Visualization Output

**User Story:** As a user, I want publication-quality plots, so that I can include them in reports and presentations.

#### Acceptance Criteria

1. WHEN generating plots, THE CLI SHALL create a cost distribution histogram showing simulation results
2. WHEN generating plots, THE CLI SHALL create a cost breakdown chart showing routing vs lead vs delegation costs
3. WHEN generating sensitivity plots, THE CLI SHALL create charts showing cost vs parameter value with error bars
4. THE CLI SHALL save all plots to the output/ directory (or specified --output-dir)
5. THE CLI SHALL generate plots with clear labels, legends, and titles
6. THE CLI SHALL use a consistent visual style across all plot types

### Requirement 9: Data Type Definitions

**User Story:** As a developer, I want well-defined data types, so that the codebase is maintainable and type-safe.

#### Acceptance Criteria

1. THE types module SHALL define dataclasses for: SimulationResult, QuerySimulation, CostBreakdown, Profile, PricingConfig
2. THE types module SHALL define enums for: QueryType (visual, code, research), Complexity (simple, moderate, complex), ModelName
3. THE types module SHALL include type hints for all public interfaces
4. FOR ALL dataclasses, serializing to JSON then deserializing SHALL produce an equivalent object (round-trip property)

### Requirement 10: Historical Data Import

**User Story:** As a user, I want to import historical usage logs, so that I can create profiles based on actual usage patterns.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor import <logfile>`, THE CLI SHALL parse JSON log entries
2. WHEN importing logs, THE CLI SHALL extract query_type, complexity, routed_to, delegated, and token counts
3. WHEN import completes, THE CLI SHALL generate a profile JSON file with computed distributions
4. IF the log file format is invalid, THEN THE CLI SHALL display a descriptive error message

### Requirement 11: Reproducible Simulations

**User Story:** As a user, I want reproducible simulation results, so that I can verify and share my analyses.

#### Acceptance Criteria

1. WHEN the --seed flag is provided, THE Simulator SHALL use the specified random seed
2. WHEN the same seed is used, THE Simulator SHALL produce identical results across runs
3. WHEN displaying results, THE CLI SHALL show the random seed used (auto-generated if not specified)

### Requirement 12: Export Results

**User Story:** As a user, I want to export simulation results, so that I can use them in other tools and reports.

#### Acceptance Criteria

1. WHEN the --export flag is provided, THE CLI SHALL save detailed results to a JSON file
2. THE exported JSON SHALL include all simulation parameters, summary statistics, and per-iteration results
3. WHEN the --export-csv flag is provided, THE CLI SHALL save summary statistics to a CSV file


### Requirement 13: User Tier Modeling

**User Story:** As a business analyst, I want to model different user tiers (paid vs free), so that I can predict costs and revenue based on user mix.

#### Acceptance Criteria

1. THE Profile SHALL support defining user tiers with different usage patterns (free, paid, enterprise)
2. WHEN simulating, THE Simulator SHALL sample user tier based on the profile's tier distribution
3. WHEN simulating a free user, THE Simulator SHALL apply usage limits (daily query cap, complexity restrictions)
4. WHEN simulating a paid user, THE Simulator SHALL apply tier-specific usage patterns and limits
5. THE CLI SHALL load tier definitions from config/tiers.json

### Requirement 14: Revenue Prediction

**User Story:** As a business analyst, I want to predict revenue alongside costs, so that I can calculate profitability and margins.

#### Acceptance Criteria

1. THE Profile SHALL support defining pricing plans with monthly subscription fees per tier
2. WHEN simulation completes, THE Cost_Calculator SHALL compute total revenue based on user counts and subscription fees
3. WHEN simulation completes, THE CLI SHALL display revenue statistics alongside cost statistics
4. WHEN simulation completes, THE CLI SHALL compute and display profit margin (revenue - costs) / revenue
5. THE CLI SHALL display break-even analysis showing minimum paid users needed for profitability

### Requirement 15: User Growth Projection

**User Story:** As a business analyst, I want to project costs and revenue over time with user growth, so that I can plan for scaling.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor forecast <profile> --months <N>`, THE CLI SHALL project costs and revenue over N months
2. WHEN forecasting, THE Simulator SHALL apply growth rates to user counts per tier
3. THE Profile SHALL support defining monthly growth rates per user tier
4. WHEN forecast completes, THE CLI SHALL display month-by-month projections of users, costs, revenue, and profit
5. WHEN forecast completes, THE CLI SHALL generate a time-series chart showing projected financials

### Requirement 16: Scenario Comparison for Business Planning

**User Story:** As a business analyst, I want to compare different business scenarios, so that I can evaluate pricing strategies and growth plans.

#### Acceptance Criteria

1. WHEN comparing profiles with revenue data, THE CLI SHALL include revenue and profit metrics in comparison tables
2. WHEN comparing profiles, THE CLI SHALL highlight the most profitable scenario
3. THE CLI SHALL support comparing scenarios with different tier distributions and pricing plans


### Requirement 17: Caching Impact Analysis

**User Story:** As a cost analyst, I want to model the impact of caching, so that I can estimate cost savings from prompt caching strategies.

#### Acceptance Criteria

1. THE Profile SHALL support defining cache hit rates for different query types
2. WHEN a cache hit occurs, THE Cost_Calculator SHALL apply reduced pricing (typically 50% or configurable)
3. WHEN simulation completes, THE CLI SHALL display cost savings from caching
4. THE CLI SHALL support sensitivity analysis on cache hit rate parameter

### Requirement 18: Error and Retry Cost Modeling

**User Story:** As a cost analyst, I want to model error rates and retries, so that I can account for real-world failure scenarios.

#### Acceptance Criteria

1. THE Profile SHALL support defining error rates per model
2. WHEN an error occurs in simulation, THE Simulator SHALL model retry behavior with associated costs
3. WHEN simulation completes, THE CLI SHALL display costs attributed to retries
4. THE Cost_Calculator SHALL include wasted tokens from failed requests in cost calculations

### Requirement 19: Latency-Cost Tradeoff Analysis

**User Story:** As a cost analyst, I want to understand latency vs cost tradeoffs, so that I can optimize for both performance and cost.

#### Acceptance Criteria

1. THE config/pricing.json SHALL support latency estimates per model
2. WHEN simulation completes, THE CLI SHALL display estimated latency statistics alongside costs
3. WHEN comparing profiles, THE CLI SHALL show latency vs cost tradeoffs
4. THE CLI SHALL support filtering models by maximum acceptable latency

### Requirement 20: Token Budget Constraints

**User Story:** As a cost analyst, I want to set budget constraints, so that I can see how limits affect service quality.

#### Acceptance Criteria

1. WHEN the --budget flag is provided, THE Simulator SHALL cap daily costs at the specified amount
2. WHEN budget is exceeded, THE Simulator SHALL track rejected queries
3. WHEN simulation completes with budget constraints, THE CLI SHALL display rejection rate and service degradation metrics
4. THE CLI SHALL support sensitivity analysis on budget parameter

### Requirement 21: Model Fallback Chains

**User Story:** As a cost analyst, I want to model fallback behavior, so that I can account for model unavailability scenarios.

#### Acceptance Criteria

1. THE config/routing.json SHALL support defining fallback chains per model
2. WHEN a model is unavailable (based on availability rate), THE Simulator SHALL route to fallback model
3. WHEN fallback occurs, THE Cost_Calculator SHALL use the fallback model's pricing
4. THE CLI SHALL display fallback frequency and associated cost impact

### Requirement 22: Batch Processing Mode

**User Story:** As a user, I want to run multiple simulations in batch, so that I can automate analysis workflows.

#### Acceptance Criteria

1. WHEN a user runs `cost-predictor batch <config-file>`, THE CLI SHALL run multiple simulations defined in the batch config
2. THE batch config SHALL support specifying multiple profiles and parameter variations
3. WHEN batch completes, THE CLI SHALL generate a summary report across all simulations
4. THE CLI SHALL support parallel execution of batch simulations for performance
