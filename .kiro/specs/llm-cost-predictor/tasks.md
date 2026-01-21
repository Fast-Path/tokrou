# Implementation Plan: Multi-Model LLM Cost Predictor

## Overview

This implementation plan breaks down the Multi-Model LLM Cost Predictor into discrete coding tasks. The approach follows a CLI-first strategy: implement and test core logic through the CLI before building the web interface. This validates the cost model accuracy early and ensures the web interface is just a wrapper around proven functionality.

## Tasks

- [x] 1. Project setup and configuration infrastructure
  - Initialize uv project with pyproject.toml
  - Create directory structure (backend/core, backend/api, backend/cli, config, tests)
  - Set up configuration file schemas and validation
  - _Requirements: 1.1, 1.6, 1.7, 1.8_

- [ ]* 1.1 Write property tests for configuration validation
  - **Property 2: Distribution Validation**
  - **Property 3: Range Validation**
  - **Validates: Requirements 1.7, 1.8**

- [x] 2. Implement core domain types
  - Create enums for QueryType, Complexity, ModelName
  - Create dataclasses for CostBreakdown, UsageProfile, SimulationResult
  - Implement validation methods for UsageProfile
  - _Requirements: 1.2, 1.3, 1.4, 1.5_

- [x] 2.1 Write property tests for domain types

  - **Property 1: Configuration Loading Completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

- [x] 3. Create configuration templates
  - Write config/profiles/conservative.yaml with pessimistic assumptions
  - Write config/profiles/baseline.yaml with expected usage patterns
  - Write config/profiles/optimistic.yaml with best-case assumptions
  - Write config/pricing.yaml with current model pricing
  - Write config/token_estimation.yaml with token estimation rules
  - Write config/simulation.yaml with simulation parameters
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_

- [x] 4. Implement TokenEstimator
  - Create TokenEstimator class with configuration loading
  - Implement estimate_classifier_tokens method
  - Implement estimate_input_tokens with range sampling
  - Implement estimate_output_tokens with multipliers
  - Implement estimate_total_tokens with system prompts and wrong model penalty
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [ ]* 4.1 Write property tests for TokenEstimator
  - **Property 4: Token Estimation Includes System Prompts**
  - **Property 5: Token Sampling Within Configured Ranges**
  - **Property 6: Output Token Multiplier Relationship**
  - **Property 7: Wrong Model Penalty Application**
  - **Property 9: Deterministic Estimation with Seed**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7**

- [ ]* 4.2 Write unit tests for TokenEstimator edge cases
  - Test classifier token estimation returns positive value
  - Test delegation token estimation includes both components
  - _Requirements: 2.5, 2.6_

- [ ] 5. Implement CostPredictor
  - Create CostPredictor class with pricing config and TokenEstimator
  - Implement _calculate_routing_cost method
  - Implement _select_lead_model with routing accuracy simulation
  - Implement _calculate_lead_cost with misrouting correction
  - Implement delegation rules mapping (DELEGATION_RULES dict)
  - Implement _get_delegation_target method
  - Implement _calculate_delegation_cost with context reinjection
  - Implement predict_query_cost orchestration method
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 5.1 Write property tests for CostPredictor
  - **Property 10: Routing Cost Calculation**
  - **Property 11: Lead Model Cost Calculation**
  - **Property 12: Misrouting Increases Cost**
  - **Property 13: Routing Accuracy Probability**
  - **Property 14: Medium Complexity Delegation Probability**
  - **Property 15: Complex Complexity Always Delegates**
  - **Property 16: Complex Second Delegation Probability**
  - **Property 17: Cost Breakdown Completeness**
  - **Property 18: Total Cost Equals Sum of Components**
  - **Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.9, 3.10**

- [ ]* 5.2 Write unit tests for CostPredictor
  - Test query type to model mapping (VISUAL→Gemini, CODE→Coder, RESEARCH→Grok)
  - Test delegation rules (Gemini→Coder/Grok, Coder→Grok, Grok→Coder)
  - Test invalid delegation prevention
  - _Requirements: 3.2, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Checkpoint - Ensure core cost model tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement Simulator
  - Create Simulator class with CostPredictor and simulation config
  - Implement _simulate_single_day method with query generation
  - Implement _simulate_single_run method with daily iteration
  - Implement _aggregate_results method with statistics calculation
  - Implement run_simulation orchestration method with seed support
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

- [ ]* 7.1 Write property tests for Simulator
  - **Property 19: Simulation Executes Correct Number of Runs**
  - **Property 20: Simulation Executes Correct Number of Days**
  - **Property 21: Simulation Generates Correct Number of Queries**
  - **Property 22: Query Type Distribution Convergence**
  - **Property 23: Complexity Distribution Convergence**
  - **Property 24: Simulation Result Completeness**
  - **Property 25: Cost Breakdown Shares Sum to One**
  - **Property 26: Monthly Estimate Calculation**
  - **Property 9: Deterministic Estimation with Seed** (simulation reproducibility)
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9**

- [ ] 8. Implement SensitivityAnalyzer
  - Create SensitivityAnalyzer class with Simulator and sensitivity config
  - Implement analyze_routing_accuracy method
  - Implement analyze_query_volume method
  - Implement analyze_complexity_distribution method
  - Implement run_full_analysis orchestration method
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ]* 8.1 Write property tests for SensitivityAnalyzer
  - **Property 27: Sensitivity Analysis Parameter Coverage**
  - **Property 28: Sensitivity Delta Calculation**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.5, 6.6**

- [ ]* 8.2 Write unit tests for SensitivityAnalyzer
  - Test baseline is included in results
  - Test custom parameter ranges are used
  - _Requirements: 6.4, 6.7_

- [ ] 9. Implement ProfileManager
  - Create ProfileManager class with profiles directory path
  - Implement list_profiles method
  - Implement load_profile method with YAML parsing
  - Implement save_profile method with validation and overwrite protection
  - Implement delete_profile method with built-in profile protection
  - Implement profile_exists and is_built_in helper methods
  - Implement validate_profile method
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7_

- [ ]* 9.1 Write property tests for ProfileManager
  - **Property 43: Profile Save and Load Round Trip**
  - **Property 44: Built-in Profile Protection**
  - **Property 45: Profile Overwrite Protection**
  - **Property 46: Profile Validation Before Save**
  - **Property 47: Profile List Completeness**
  - **Validates: Requirements 18.1, 18.2, 18.5, 18.6, 18.7**

- [ ] 10. Checkpoint - Ensure all core logic tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement CLI interface
  - Create CLI main.py with Click group
  - Implement simulate command with profile loading and parameter overrides
  - Implement sensitivity command with JSON output
  - Implement compare command with multiple profiles
  - Implement report command with format selection (text, JSON, markdown)
  - Implement create-profile command with interactive mode
  - Implement delete-profile command with confirmation
  - Implement list-profiles command
  - Add formatted text output for simulate command
  - Add error handling with descriptive messages
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 18.1, 18.3_

- [ ]* 11.1 Write property tests for CLI
  - **Property 29: CLI Parameter Override Application**
  - **Property 30: CLI Output Completeness**
  - **Property 31: CLI Compare Includes All Profiles**
  - **Property 32: CLI Output File Creation**
  - **Validates: Requirements 7.3, 7.4, 7.6, 7.9**

- [ ]* 11.2 Write unit tests for CLI commands
  - Test simulate with profile name loads from config/profiles/
  - Test simulate with profile file loads from specified path
  - Test sensitivity outputs valid JSON
  - Test report generates all formats (text, JSON, markdown)
  - Test invalid config displays error messages
  - Test create-profile saves to config/profiles/
  - Test delete-profile removes file and prevents built-in deletion
  - _Requirements: 7.1, 7.2, 7.5, 7.7, 7.8, 18.1, 18.3_

- [ ] 12. Checkpoint - Test CLI end-to-end
  - Run CLI simulate command with baseline profile
  - Verify output contains monthly estimate, P95, P99, breakdown, metrics
  - Run CLI sensitivity command and verify JSON output
  - Run CLI compare command with multiple profiles
  - Run CLI create-profile and delete-profile commands
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement Flask API
  - Create Flask app in api/app.py with CORS configuration
  - Implement POST /api/simulate endpoint with request validation
  - Implement POST /api/sensitivity endpoint
  - Implement GET /api/profiles endpoint
  - Implement GET /api/profiles/<name> endpoint
  - Implement POST /api/profiles endpoint with validation
  - Implement DELETE /api/profiles/<name> endpoint with built-in protection
  - Implement GET /api/config/pricing endpoint
  - Add error handling with 400 status for invalid requests
  - Add JSON response formatting
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 18.2, 18.4, 18.9_

- [ ]* 13.1 Write property tests for API
  - **Property 33: API Simulate Response Structure**
  - **Property 34: API Sensitivity Response Structure**
  - **Property 35: API Validation Error Responses**
  - **Property 36: API JSON Response Format**
  - **Validates: Requirements 8.1, 8.2, 8.5, 8.6**

- [ ]* 13.2 Write unit tests for API endpoints
  - Test GET /api/profiles returns list of profile names
  - Test GET /api/config/pricing returns pricing config
  - Test POST /api/profiles saves profile
  - Test DELETE /api/profiles/<name> removes profile
  - Test DELETE prevents built-in profile deletion
  - _Requirements: 8.3, 8.4, 18.2, 18.4, 18.9_

- [ ] 14. Implement validation script
  - Create tests/validate_estimates.py script
  - Load hand-crafted test queries (5-10 spanning all types/complexities)
  - Implement real API execution functions (Gemini, Cerebras, xAI)
  - Implement token usage comparison logic
  - Implement MAPE calculation
  - Add failure condition for MAPE > 30%
  - Add per-query error reporting
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

- [ ]* 14.1 Write property test for MAPE calculation
  - **Property 42: MAPE Calculation Correctness**
  - **Validates: Requirements 13.5**

- [ ]* 14.2 Write unit tests for validation script
  - Test validation script exists at correct path
  - Test test queries load correctly
  - Test comparison logic works
  - Test MAPE calculation with known values
  - Test script fails when MAPE > 30%
  - Test per-query error reporting
  - _Requirements: 13.1, 13.2, 13.4, 13.6, 13.7_

- [ ] 15. Checkpoint - Ensure backend is complete
  - Run full test suite (unit + property tests)
  - Test CLI commands work end-to-end
  - Test API endpoints return valid responses
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement React frontend - ConfigEditor component
  - Create ConfigEditor.jsx with state management
  - Add sliders for query distribution with live sum validation
  - Add sliders for complexity distribution with live sum validation
  - Add input for routing accuracy with range validation
  - Add input for queries per day
  - Add profile name input for saving
  - Add load/save/delete buttons
  - Add confirmation dialogs for overwrite and delete
  - Add validation error display
  - Prevent deletion of built-in profiles
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 18.8, 18.9_

- [ ]* 16.1 Write property tests for ConfigEditor validation logic
  - **Property 37: Config Editor Distribution Sum Validation**
  - **Property 38: Config Editor Profile Population**
  - **Property 39: Config Editor Validation State**
  - **Validates: Requirements 9.3, 9.6, 9.7**

- [ ] 17. Implement React frontend - SimulationRunner component
  - Create SimulationRunner.jsx with state management
  - Add profile selection dropdown
  - Add run simulation button
  - Add progress indicator
  - Implement API call to POST /api/simulate
  - Pass results to CostDashboard component
  - Add sensitivity analysis button
  - Implement API call to POST /api/sensitivity
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 12.4_

- [ ]* 17.1 Write unit tests for SimulationRunner
  - Test API call is invoked with correct parameters
  - Test results are passed to dashboard
  - Test sensitivity analysis trigger
  - _Requirements: 10.3, 10.5, 12.4_

- [ ] 18. Implement React frontend - CostDashboard component
  - Create CostDashboard.jsx with props for simulation results
  - Add metric cards for monthly estimate, P95, P99
  - Add pie chart for cost breakdown (routing, lead, delegation)
  - Add histogram for daily cost distribution
  - Add comparison table for multiple scenarios
  - Implement monetary value formatting (2 decimal places)
  - Implement percentage value formatting (1 decimal place)
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ]* 18.1 Write property tests for formatting functions
  - **Property 40: Monetary Value Formatting**
  - **Property 41: Percentage Value Formatting**
  - **Validates: Requirements 11.4, 11.5**

- [ ] 19. Implement React frontend - SensitivityChart component
  - Create SensitivityChart.jsx with props for sensitivity results
  - Add line chart for cost vs. parameter value
  - Add tornado chart for impact magnitude
  - Add interactive tooltips with exact values
  - _Requirements: 12.1, 12.2, 12.3_

- [ ] 20. Implement React frontend - App integration
  - Create App.jsx with component layout
  - Wire ConfigEditor, SimulationRunner, CostDashboard, SensitivityChart
  - Add state management for profiles and results
  - Add API integration for profile management
  - Add export functionality (JSON, CSV)
  - _Requirements: 15.4, 15.5_

- [ ]* 20.1 Write unit tests for export functionality
  - Test JSON export produces valid JSON
  - Test CSV export produces valid CSV
  - _Requirements: 15.4, 15.5_

- [ ] 21. Create documentation
  - Write README.md with installation instructions using uv
  - Add usage examples for CLI commands
  - Add usage examples for web interface
  - Write docs/cost_model.md explaining calculation logic
  - Write docs/assumptions.md documenting assumptions and sources
  - Write docs/calibration.md with token estimate update instructions
  - Add inline code comments explaining cost model logic
  - Add comments to configuration files
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7_

- [ ] 22. Create Makefile for common tasks
  - Add run-cli target for CLI execution
  - Add run-web target for Flask server
  - Add run-frontend target for React dev server
  - Add test target for running all tests
  - Add validate target for running validation script
  - Add format target for code formatting
  - Add lint target for code linting

- [ ] 23. Final integration testing
  - Test complete workflow: create profile → run simulation → view results
  - Test CLI and web interface produce consistent results
  - Test configuration changes update all calculations
  - Test profile management (create, save, delete) works in both CLI and web
  - Verify all documentation is accurate
  - Run validation script against real APIs (if API keys available)

- [ ] 24. Final checkpoint - Project complete
  - All tests passing (unit + property + integration)
  - CLI works end-to-end
  - Web interface functional
  - Documentation complete
  - Validation script ready for calibration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- CLI-first approach validates core logic before building web interface
- Profile management enables scenario exploration without code changes
