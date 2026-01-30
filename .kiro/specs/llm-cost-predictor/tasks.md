# Implementation Tasks: LLM Cost Predictor

## Phase 1: Project Setup and Core Types

- [x] 1. Project initialization
  - [x] 1.1 Create project structure (directories: cli.py, core/, config/, output/, tests/)
  - [x] 1.2 Create pyproject.toml with dependencies (click, numpy, matplotlib, hypothesis)
  - [x] 1.3 Configure uv package manager

- [x] 2. Implement core type definitions (core/types.py)
  - [x] 2.1 Define enums (QueryType, Complexity, UserTier)
  - [x] 2.2 Define dataclasses (ModelPricing, TokenEstimate, QuerySimulation, CostBreakdown, TierConfig, Profile, SimulationResult)
  - [x] 2.3 Implement JSON serialization methods (to_json, from_json) for SimulationResult
  - [x] 2.4 Write property test for dataclass round-trip serialization (Property 13)

## Phase 2: Configuration Management

- [x] 3. Implement configuration loader (core/config.py)
  - [x] 3.1 Implement ConfigLoader class with __init__ method
  - [x] 3.2 Implement load_pricing method
  - [x] 3.3 Implement load_profile method
  - [x] 3.4 Implement load_token_estimation method
  - [x] 3.5 Implement load_routing method
  - [x] 3.6 Implement load_tiers method
  - [x] 3.7 Implement _load_json helper with error handling
  - [x] 3.8 Write unit tests for config loading (valid and invalid files)

- [x] 4. Create default configuration files
  - [x] 4.1 Create config/pricing.json with model pricing data
  - [x] 4.2 Create config/profiles/default.json with default usage profile
  - [x] 4.3 Create config/token_estimation.json with token rules
  - [x] 4.4 Create config/routing.json with routing configuration
  - [x] 4.5 Create config/tiers.json with user tier definitions

## Phase 3: Cost Calculation and Token Estimation

- [x] 5. Implement token estimator (core/token_estimator.py)
  - [x] 5.1 Implement TokenEstimator class with __init__ method
  - [x] 5.2 Implement estimate method for token calculation
  - [x] 5.3 Write unit tests for token estimation by query type (Property 9)
  - [x] 5.4 Write property test for complexity multiplier application (Property 10)
  - [x] 5.5 Write property test for delegation token overhead (Property 11)

- [ ] 6. Implement cost calculator (core/cost_calculator.py)
  - [ ] 6.1 Implement CostCalculator class with __init__ method
  - [ ] 6.2 Implement calculate method for cost computation
  - [ ] 6.3 Implement estimate_latency method
  - [ ] 6.4 Implement calculate_break_even method
  - [ ] 6.5 Write property test for cost formula correctness (Property 12)
  - [ ] 6.6 Write property test for caching cost reduction (Property 23)
  - [ ] 6.7 Write property test for retry cost inclusion (Property 24)
  - [ ] 6.8 Write property test for latency calculation (Property 25)

## Phase 4: Monte Carlo Simulator

- [ ] 7. Implement simulator core (core/simulator.py)
  - [ ] 7.1 Implement Simulator class with __init__ method
  - [ ] 7.2 Implement run method (main simulation loop)
  - [ ] 7.3 Implement _simulate_day method
  - [ ] 7.4 Implement _simulate_query method
  - [ ] 7.5 Implement _sample_tier method
  - [ ] 7.6 Implement _sample_query_type method
  - [ ] 7.7 Implement _sample_complexity method (with tier restrictions)
  - [ ] 7.8 Implement _get_route method
  - [ ] 7.9 Implement _build_delegation_chain method
  - [ ] 7.10 Implement _simulate_errors method
  - [ ] 7.11 Implement _aggregate_results method

- [ ] 8. Write property tests for simulator
  - [ ] 8.1 Write property test for iteration count consistency (Property 1)
  - [ ] 8.2 Write property test for query type distribution convergence (Property 2)
  - [ ] 8.3 Write property test for complexity distribution convergence (Property 3)
  - [ ] 8.4 Write property test for routing accuracy convergence (Property 4)
  - [ ] 8.5 Write property test for incorrect routing cost penalty (Property 5)
  - [ ] 8.6 Write property test for delegation probability convergence (Property 6)
  - [ ] 8.7 Write property test for delegation chain cost inclusion (Property 7)
  - [ ] 8.8 Write property test for statistics calculation correctness (Property 8)
  - [ ] 8.9 Write property test for seed determinism (Property 15)
  - [ ] 8.10 Write property test for tier distribution convergence (Property 17)
  - [ ] 8.11 Write property test for tier usage limits (Property 18)
  - [ ] 8.12 Write property test for budget constraint enforcement (Property 26)

## Phase 5: Revenue and Business Metrics

- [ ] 9. Implement revenue calculations
  - [ ] 9.1 Add revenue calculation logic to _aggregate_results method
  - [ ] 9.2 Write property test for revenue calculation correctness (Property 19)
  - [ ] 9.3 Write property test for profit margin calculation (Property 20)
  - [ ] 9.4 Write property test for break-even calculation (Property 21)
  - [ ] 9.5 Write property test for growth rate application (Property 22)

## Phase 6: Visualization

- [ ] 10. Implement plot generator (core/plots.py)
  - [ ] 10.1 Implement PlotGenerator class with __init__ method
  - [ ] 10.2 Implement cost_distribution method (histogram)
  - [ ] 10.3 Implement cost_breakdown method (pie chart)
  - [ ] 10.4 Implement comparison method (bar chart)
  - [ ] 10.5 Implement sensitivity method (line chart with error bars)
  - [ ] 10.6 Implement forecast method (time series chart)
  - [ ] 10.7 Write integration tests for plot generation

## Phase 7: CLI Commands

- [ ] 11. Implement CLI entry point (cli.py)
  - [ ] 11.1 Implement main cli group with Click (already exists, needs config loader integration)
  - [ ] 11.2 Implement simulate command
  - [ ] 11.3 Implement compare command
  - [ ] 11.4 Implement sensitivity command
  - [ ] 11.5 Implement forecast command
  - [ ] 11.6 Implement import command (log importer)
  - [ ] 11.7 Implement batch command
  - [ ] 11.8 Add error handling and user-friendly messages

- [ ] 12. Write integration tests for CLI
  - [ ] 12.1 Test simulate command with various flags
  - [ ] 12.2 Test compare command with multiple profiles
  - [ ] 12.3 Test sensitivity command with different parameters
  - [ ] 12.4 Test forecast command
  - [ ] 12.5 Test import command with sample log file
  - [ ] 12.6 Test error handling (missing files, invalid JSON)

## Phase 8: Log Importer

- [ ] 13. Implement log importer (core/importer.py)
  - [ ] 13.1 Implement LogImporter class
  - [ ] 13.2 Implement import_logs method
  - [ ] 13.3 Implement _parse_logs method (handle both JSON array and JSONL)
  - [ ] 13.4 Write property test for log import distribution accuracy (Property 14)
  - [ ] 13.5 Write unit tests for log parsing edge cases

## Phase 9: Export Functionality

- [ ] 14. Implement export handlers (core/export.py)
  - [ ] 14.1 Implement JSON export functionality
  - [ ] 14.2 Implement CSV export functionality
  - [ ] 14.3 Write property test for export completeness (Property 16)
  - [ ] 14.4 Write integration tests for export file formats

## Phase 10: Advanced Features

- [ ] 15. Implement fallback chain logic
  - [ ] 15.1 Add model availability simulation to _simulate_query
  - [ ] 15.2 Implement fallback routing logic
  - [ ] 15.3 Write property test for fallback chain activation (Property 27)

- [ ] 16. Implement batch processing
  - [ ] 16.1 Create batch configuration schema
  - [ ] 16.2 Implement parallel execution support
  - [ ] 16.3 Implement batch summary report generation
  - [ ] 16.4 Write integration tests for batch mode

## Phase 11: Documentation and Polish

- [ ] 17. Create documentation
  - [ ] 17.1 Write README.md with installation and usage instructions
  - [ ] 17.2 Create example configuration files with comments
  - [ ] 17.3 Add docstrings to all public methods
  - [ ] 17.4 Create usage examples for each CLI command

- [ ] 18. Final testing and validation
  - [ ] 18.1 Run full property test suite (minimum 100 examples each)
  - [ ] 18.2 Run integration tests for all CLI commands
  - [ ] 18.3 Test with provided sample log data
  - [ ] 18.4 Validate plot quality and formatting
  - [ ] 18.5 Perform end-to-end workflow test

- [ ] 19. Performance optimization
  - [ ] 19.1 Profile simulation performance
  - [ ] 19.2 Optimize hot paths if needed
  - [ ] 19.3 Add progress indicators for long-running simulations
