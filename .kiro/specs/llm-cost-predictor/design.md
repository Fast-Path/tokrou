# Design Document

## Overview

The Multi-Model LLM Cost Predictor is a cost analysis system for predicting operational expenses of a multi-model LLM orchestration architecture. The system uses Monte Carlo simulation to model query routing, delegation patterns, and token usage across three specialized models (Gemini for visual tasks, Coder for code generation, Grok for research) plus a classifier for routing.

The architecture follows a strict separation of concerns:
- **Core logic** (`backend/core/`): Pure Python calculation engine with no I/O dependencies
- **CLI interface** (`backend/cli/`): Command-line wrapper using Click
- **API layer** (`backend/api/`): Flask REST API for web consumption
- **Web interface** (`frontend/`): React-based interactive dashboard

All parameters (pricing, token estimates, usage patterns) are externalized in YAML configuration files, enabling scenario testing without code changes.

## Architecture

### Project Setup with uv

The project uses `uv` for fast, reliable Python package management:

**Installation:**
```bash

# Initialize project
uv init

# Add dependencies
uv add flask numpy pyyaml click hypothesis pytest pytest-cov

# Add dev dependencies
uv add --dev black ruff mypy

# Run commands
uv run python -m backend.cli.main simulate --profile baseline
uv run flask --app backend.api.app run
uv run pytest tests/
```

**Benefits of uv:**
- 10-100x faster than pip
- Deterministic dependency resolution
- Built-in virtual environment management
- Compatible with pip and requirements.txt

### System Components

**Component Diagram:**
```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (React)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ ConfigEditor │  │ SimRunner    │  │ CostDashboard│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/JSON
┌────────────────────────▼────────────────────────────────────┐
│                      Flask API Layer                        │
│  POST /api/simulate  │  POST /api/sensitivity               │
│  GET /api/profiles   │  POST /api/profiles                  │
│  DELETE /api/profiles/<name>                                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      CLI Interface                          │
│  simulate  │  sensitivity  │  compare  │  report            │
│  create-profile  │  delete-profile  │  list-profiles        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Core Calculation Engine                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │TokenEstimator│  │CostPredictor │  │  Simulator   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Sensitivity  │  │ProfileManager│  │    Types     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Configuration Files (YAML)                 │
│  profiles/  │  pricing.yaml  │  token_estimation.yaml       │
└─────────────────────────────────────────────────────────────┘
```

**Project Structure:**

```
cost-predictor/
├── config/
│   ├── profiles/
│   │   ├── conservative.yaml
│   │   ├── baseline.yaml
│   │   └── optimistic.yaml
│   ├── pricing.yaml
│   ├── token_estimation.yaml
│   └── simulation.yaml
│
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py              # Enums, dataclasses
│   │   ├── token_estimator.py    # Token prediction logic
│   │   ├── cost_predictor.py     # Single-query cost calculation
│   │   ├── simulator.py          # Monte Carlo simulation engine
│   │   ├── sensitivity.py        # Parameter sensitivity analysis
│   │   └── profile_manager.py    # Profile CRUD operations
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                # Flask app
│   │   └── routes.py             # API endpoints
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py               # CLI interface
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ConfigEditor.jsx      # Edit usage profiles
│   │   │   ├── SimulationRunner.jsx  # Trigger simulations
│   │   │   ├── CostDashboard.jsx     # Visualize results
│   │   │   └── SensitivityChart.jsx  # Sensitivity analysis charts
│   │   ├── App.jsx
│   │   └── index.js
│   └── package.json
│
├── tests/
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_token_estimator.py
│   │   ├── test_cost_predictor.py
│   │   ├── test_simulator.py
│   │   ├── test_sensitivity.py
│   │   ├── test_profile_manager.py
│   │   ├── test_cli.py
│   │   └── test_api.py
│   ├── property/
│   │   ├── test_config_properties.py
│   │   ├── test_token_properties.py
│   │   ├── test_cost_properties.py
│   │   ├── test_simulation_properties.py
│   │   ├── test_sensitivity_properties.py
│   │   ├── test_profile_properties.py
│   │   ├── test_cli_properties.py
│   │   ├── test_api_properties.py
│   │   └── test_ui_properties.py
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   └── test_config_loading.py
│   └── validate_estimates.py
│
├── pyproject.toml              # uv project configuration
├── README.md
└── Makefile                    # Common tasks (run-cli, run-web, test)
```

### Technology Stack

- **Backend**: Python 3.10+, Flask, NumPy, PyYAML, Click
- **Package Manager**: uv for Python dependency management
- **Frontend**: React 18+, functional components with hooks
- **Testing**: pytest for unit/integration tests, property-based testing for core logic
- **Configuration**: YAML files for all parameters

## Components and Interfaces

### Core Types (`backend/core/types.py`)

Defines domain enums and dataclasses:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class QueryType(Enum):
    VISUAL = "visual"
    CODE = "code"
    RESEARCH = "research"

class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ModelName(Enum):
    GEMINI = "gemini"
    CODER = "coder"
    GROK = "grok"
    CLASSIFIER = "classifier"

@dataclass
class CostBreakdown:
    routing_cost: float
    lead_cost: float
    delegation_cost: float
    total_cost: float
    query_type: QueryType
    complexity: Complexity
    was_routed_correctly: bool
    num_delegations: int

@dataclass
class UsageProfile:
    name: str
    description: str
    query_distribution: Dict[str, float]  # {visual: 0.2, code: 0.3, research: 0.5}
    complexity_distribution: Dict[str, float]  # {simple: 0.6, medium: 0.3, complex: 0.1}
    routing_accuracy: float
    delegation_accuracy: float
    queries_per_day: int

@dataclass
class SimulationResult:
    mean_daily_cost: float
    median_daily_cost: float
    p95_daily_cost: float
    p99_daily_cost: float
    monthly_estimate: float
    std_dev: float
    routing_cost_share: float
    lead_cost_share: float
    delegation_cost_share: float
    avg_delegations_per_query: float
    realized_routing_accuracy: float
```

### Token Estimator (`backend/core/token_estimator.py`)

Predicts token usage for queries based on type, complexity, and model.

**Interface:**
```python
class TokenEstimator:
    def __init__(self, config: dict):
        """Initialize with token_estimation.yaml config"""
        
    def estimate_classifier_tokens(self, query_text: str = None) -> int:
        """Estimate tokens for classifier routing decision"""
        
    def estimate_input_tokens(
        self, 
        query_type: QueryType, 
        complexity: Complexity,
        model: ModelName,
        rng: np.random.Generator
    ) -> int:
        """Sample input tokens from configured range"""
        
    def estimate_output_tokens(
        self,
        input_tokens: int,
        complexity: Complexity
    ) -> int:
        """Calculate output tokens using complexity multiplier"""
        
    def estimate_total_tokens(
        self,
        query_type: QueryType,
        complexity: Complexity,
        model: ModelName,
        is_correct_model: bool,
        rng: np.random.Generator
    ) -> tuple[int, int]:
        """Returns (input_tokens, output_tokens) including system prompt"""
```

**Key Logic:**
- System prompt tokens added to all estimates
- Input tokens sampled uniformly from configured ranges
- Output tokens = input_tokens × complexity_multiplier
- Wrong model penalty applied when `is_correct_model=False`

### Cost Predictor (`backend/core/cost_predictor.py`)

Calculates cost for a single query including routing, lead model, and delegation.

**Interface:**
```python
class CostPredictor:
    def __init__(
        self,
        pricing_config: dict,
        token_estimator: TokenEstimator
    ):
        """Initialize with pricing and token estimator"""
        
    def predict_query_cost(
        self,
        query_type: QueryType,
        complexity: Complexity,
        routing_accuracy: float,
        delegation_accuracy: float,
        rng: np.random.Generator
    ) -> CostBreakdown:
        """Predict cost for a single query"""
        
    def _calculate_routing_cost(self, rng: np.random.Generator) -> float:
        """Calculate classifier invocation cost"""
        
    def _select_lead_model(
        self,
        query_type: QueryType,
        routing_accuracy: float,
        rng: np.random.Generator
    ) -> tuple[ModelName, bool]:
        """Returns (selected_model, was_correct)"""
        
    def _calculate_lead_cost(
        self,
        query_type: QueryType,
        complexity: Complexity,
        actual_lead: ModelName,
        correct_lead: ModelName,
        rng: np.random.Generator
    ) -> float:
        """Calculate lead model cost including misrouting correction"""
        
    def _calculate_delegation_cost(
        self,
        query_type: QueryType,
        complexity: Complexity,
        lead_model: ModelName,
        delegation_accuracy: float,
        original_input_tokens: int,
        rng: np.random.Generator
    ) -> tuple[float, int]:
        """Returns (delegation_cost, num_delegations)"""
```

**Key Logic - Delegation Cost Calculation:**

The delegation cost is the most complex part of the model due to context reinjection:

```python
def _calculate_delegation_cost(self, ...):
    # Determine if delegation needed
    if complexity == SIMPLE:
        return 0.0, 0
    
    needs_delegation = (
        complexity == COMPLEX or 
        (complexity == MEDIUM and rng.random() < 0.8)
    )
    
    if not needs_delegation:
        return 0.0, 0
    
    # First delegation
    tool_model = self._get_delegation_target(query_type, lead_model)
    
    # Tool execution cost
    tool_input, tool_output = self.token_estimator.estimate_total_tokens(
        query_type, complexity, tool_model, True, rng
    )
    tool_cost = self._calculate_token_cost(
        tool_model, tool_input, tool_output
    )
    
    # Context reinjection cost (CRITICAL: non-linear growth)
    # Lead must reprocess: original_query + tool_result
    context_input = original_input_tokens + tool_output
    context_output = self.token_estimator.estimate_output_tokens(
        context_input, complexity
    )
    reinjection_cost = self._calculate_token_cost(
        lead_model, context_input, context_output
    )
    
    delegation_cost = tool_cost + reinjection_cost
    num_delegations = 1
    
    # Second delegation for complex queries (30% chance)
    if complexity == COMPLEX and rng.random() < 0.3:
        # Context has grown: original + tool_1_output + tool_2_output
        # Simplified: 60% of first delegation cost
        delegation_cost += delegation_cost * 0.6
        num_delegations = 2
    
    return delegation_cost, num_delegations
```

**Delegation Rules:**
```python
DELEGATION_RULES = {
    ModelName.GEMINI: {
        QueryType.CODE: ModelName.CODER,
        QueryType.RESEARCH: ModelName.GROK
    },
    ModelName.CODER: {
        QueryType.RESEARCH: ModelName.GROK
    },
    ModelName.GROK: {
        QueryType.CODE: ModelName.CODER
    }
}
```

### Simulator (`backend/core/simulator.py`)

Runs Monte Carlo simulations to predict cost distributions.

**Interface:**
```python
class Simulator:
    def __init__(
        self,
        cost_predictor: CostPredictor,
        simulation_config: dict
    ):
        """Initialize with cost predictor and simulation config"""
        
    def run_simulation(
        self,
        profile: UsageProfile,
        num_runs: int = None,
        num_days: int = None,
        seed: int = None
    ) -> SimulationResult:
        """Run Monte Carlo simulation"""
        
    def _simulate_single_run(
        self,
        profile: UsageProfile,
        num_days: int,
        rng: np.random.Generator
    ) -> List[float]:
        """Returns list of daily costs for one run"""
        
    def _simulate_single_day(
        self,
        profile: UsageProfile,
        rng: np.random.Generator
    ) -> tuple[float, List[CostBreakdown]]:
        """Returns (day_cost, breakdowns)"""
```

**Key Logic:**
```python
def run_simulation(self, profile, num_runs, num_days, seed):
    rng = np.random.default_rng(seed)
    all_daily_costs = []
    all_breakdowns = []
    
    for run in range(num_runs):
        daily_costs = []
        for day in range(num_days):
            day_cost = 0.0
            day_breakdowns = []
            
            for _ in range(profile.queries_per_day):
                # Sample query type
                query_type = rng.choice(
                    list(QueryType),
                    p=self._distribution_to_probabilities(
                        profile.query_distribution
                    )
                )
                
                # Sample complexity
                complexity = rng.choice(
                    list(Complexity),
                    p=self._distribution_to_probabilities(
                        profile.complexity_distribution
                    )
                )
                
                # Predict cost
                breakdown = self.cost_predictor.predict_query_cost(
                    query_type,
                    complexity,
                    profile.routing_accuracy,
                    profile.delegation_accuracy,
                    rng
                )
                
                day_cost += breakdown.total_cost
                day_breakdowns.append(breakdown)
            
            daily_costs.append(day_cost)
            all_breakdowns.extend(day_breakdowns)
        
        all_daily_costs.extend(daily_costs)
    
    return self._aggregate_results(all_daily_costs, all_breakdowns)
```

### Sensitivity Analyzer (`backend/core/sensitivity.py`)

Tests how costs vary with parameter changes.

**Interface:**
```python
class SensitivityAnalyzer:
    def __init__(
        self,
        simulator: Simulator,
        sensitivity_config: dict
    ):
        """Initialize with simulator and sensitivity config"""
        
    def analyze_routing_accuracy(
        self,
        baseline_profile: UsageProfile
    ) -> Dict[float, SimulationResult]:
        """Vary routing accuracy, return results by accuracy value"""
        
    def analyze_query_volume(
        self,
        baseline_profile: UsageProfile
    ) -> Dict[int, SimulationResult]:
        """Vary queries per day, return results by volume"""
        
    def analyze_complexity_distribution(
        self,
        baseline_profile: UsageProfile
    ) -> Dict[float, SimulationResult]:
        """Vary complex query percentage, return results"""
        
    def run_full_analysis(
        self,
        baseline_profile: UsageProfile,
        parameters: List[str]
    ) -> dict:
        """Run analysis for specified parameters"""
```

### Profile Manager (`backend/core/profile_manager.py`)

Manages loading, saving, and deleting usage profiles.

**Interface:**
```python
class ProfileManager:
    BUILT_IN_PROFILES = ['conservative', 'baseline', 'optimistic']
    
    def __init__(self, profiles_dir: str = 'config/profiles'):
        """Initialize with profiles directory path"""
        
    def list_profiles(self) -> List[str]:
        """Return list of all available profile names"""
        
    def load_profile(self, name: str) -> UsageProfile:
        """Load profile by name from profiles directory"""
        
    def save_profile(
        self,
        name: str,
        profile: UsageProfile,
        overwrite: bool = False
    ) -> None:
        """
        Save profile to profiles directory
        Raises ValueError if profile exists and overwrite=False
        """
        
    def delete_profile(self, name: str) -> None:
        """
        Delete profile from profiles directory
        Raises ValueError if trying to delete built-in profile
        """
        
    def profile_exists(self, name: str) -> bool:
        """Check if profile exists"""
        
    def is_built_in(self, name: str) -> bool:
        """Check if profile is a built-in profile"""
        
    def validate_profile(self, profile: UsageProfile) -> List[str]:
        """
        Validate profile data
        Returns list of validation errors (empty if valid)
        """
```

**Key Logic:**
```python
def save_profile(self, name, profile, overwrite):
    # Validate profile first
    errors = self.validate_profile(profile)
    if errors:
        raise ValueError(f"Invalid profile: {', '.join(errors)}")
    
    # Check if exists
    profile_path = os.path.join(self.profiles_dir, f"{name}.yaml")
    if os.path.exists(profile_path) and not overwrite:
        raise ValueError(f"Profile '{name}' already exists. Use overwrite=True to replace.")
    
    # Save to YAML
    with open(profile_path, 'w') as f:
        yaml.dump(profile_to_dict(profile), f)

def delete_profile(self, name):
    # Prevent deletion of built-in profiles
    if self.is_built_in(name):
        raise ValueError(f"Cannot delete built-in profile '{name}'")
    
    # Delete file
    profile_path = os.path.join(self.profiles_dir, f"{name}.yaml")
    if not os.path.exists(profile_path):
        raise ValueError(f"Profile '{name}' not found")
    
    os.remove(profile_path)
```

### CLI Interface (`backend/cli/main.py`)

Command-line interface using Click.

**Commands:**
```python
@click.group()
def cli():
    """Multi-Model LLM Cost Predictor CLI"""

@cli.command()
@click.option('--profile', help='Profile name from config/profiles/')
@click.option('--profile-file', help='Path to custom profile YAML')
@click.option('--routing-accuracy', type=float, help='Override routing accuracy')
@click.option('--queries-per-day', type=int, help='Override queries per day')
@click.option('--runs', type=int, help='Override number of runs')
@click.option('--output', help='Output file path')
def simulate(...):
    """Run cost simulation"""

@cli.command()
@click.option('--profile', required=True)
@click.option('--output', help='Output JSON file')
def sensitivity(...):
    """Run sensitivity analysis"""

@cli.command()
@click.option('--profiles', multiple=True, required=True)
def compare(...):
    """Compare multiple profiles"""

@cli.command()
@click.option('--profile', required=True)
@click.option('--output', required=True)
@click.option('--format', type=click.Choice(['text', 'json', 'markdown']))
def report(...):
    """Generate full report"""

@cli.command()
@click.argument('name')
@click.option('--from-profile', help='Copy from existing profile')
@click.option('--interactive', is_flag=True, help='Interactive profile creation')
def create_profile(name, from_profile, interactive):
    """Create a new usage profile"""

@cli.command()
@click.argument('name')
@click.option('--force', is_flag=True, help='Skip confirmation')
def delete_profile(name, force):
    """Delete a usage profile (prevents deletion of built-in profiles)"""

@cli.command()
def list_profiles():
    """List all available profiles"""
```

### Flask API (`backend/api/routes.py`)

REST API endpoints.

**Endpoints:**
```python
@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Request: {
        "profile": {...},
        "simulation": {"runs": 100, "days": 30}
    }
    Response: {
        "projection": {...},
        "breakdown": {...},
        "metrics": {...}
    }
    """

@app.route('/api/sensitivity', methods=['POST'])
def sensitivity():
    """
    Request: {
        "baseline_profile": {...},
        "parameters_to_vary": [...]
    }
    Response: {
        "baseline": {...},
        "scenarios": {...}
    }
    """

@app.route('/api/profiles', methods=['GET'])
def list_profiles():
    """Returns: ["conservative", "baseline", "optimistic", "custom1", ...]"""

@app.route('/api/profiles/<name>', methods=['GET'])
def get_profile(name):
    """Returns: profile data for specified name"""

@app.route('/api/profiles', methods=['POST'])
def create_profile():
    """
    Request: {
        "name": "my-profile",
        "profile": {...}
    }
    Response: {"success": true, "message": "Profile saved"}
    """

@app.route('/api/profiles/<name>', methods=['DELETE'])
def delete_profile(name):
    """
    Response: {"success": true, "message": "Profile deleted"}
    Prevents deletion of built-in profiles
    """

@app.route('/api/config/pricing', methods=['GET'])
def get_pricing():
    """Returns: pricing.yaml contents"""
```

### React Components

**ConfigEditor.jsx:**
- Sliders for query distribution (visual, code, research) with live sum validation
- Sliders for complexity distribution (simple, medium, complex) with live sum validation
- Number input for routing accuracy (0.0-1.0)
- Number input for queries per day
- Profile load/save/delete buttons
- Profile name input for saving new profiles
- Confirmation dialog for overwriting existing profiles
- Confirmation dialog for deleting profiles
- Validation error display
- Prevents deletion of built-in profiles (conservative, baseline, optimistic)

**SimulationRunner.jsx:**
- Profile dropdown selector
- "Run Simulation" button (disabled if config invalid)
- Progress bar during execution
- Triggers API calls to `/api/simulate`

**CostDashboard.jsx:**
- Metric cards: Monthly Estimate, P95, P99
- Pie chart: Cost breakdown (routing, lead, delegation)
- Histogram: Daily cost distribution
- Comparison table for multiple scenarios

**SensitivityChart.jsx:**
- Line chart: Cost vs. parameter value
- Tornado chart: Impact magnitude by parameter
- Interactive tooltips with exact values

## Data Models

### Configuration File Schemas

**Profile Schema (config/profiles/*.yaml):**
```yaml
name: string
description: string
query_distribution:
  visual: float (0.0-1.0)
  code: float (0.0-1.0)
  research: float (0.0-1.0)
  # Must sum to 1.0
complexity_distribution:
  simple: float (0.0-1.0)
  medium: float (0.0-1.0)
  complex: float (0.0-1.0)
  # Must sum to 1.0
routing_accuracy: float (0.0-1.0)
delegation_accuracy: float (0.0-1.0)
queries_per_day: int (> 0)
```

**Pricing Schema (config/pricing.yaml):**
```yaml
models:
  gemini:
    input: float  # Cost per token
    output: float
  coder:
    input: float
    output: float
  grok:
    input: float
    output: float
  classifier:
    input: float
    output: float
```

**Token Estimation Schema (config/token_estimation.yaml):**
```yaml
system_prompts:
  gemini: int
  coder: int
  grok: int
  classifier: int
input_ranges:
  visual:
    simple: [int, int]
    medium: [int, int]
    complex: [int, int]
  code:
    simple: [int, int]
    medium: [int, int]
    complex: [int, int]
  research:
    simple: [int, int]
    medium: [int, int]
    complex: [int, int]
output_multipliers:
  simple: float
  medium: float
  complex: float
wrong_model_penalty: float
```

**Simulation Schema (config/simulation.yaml):**
```yaml
runs: int
days_to_simulate: int
sensitivity_analysis:
  routing_accuracy_range: [float, ...]
  query_volume_range: [int, ...]
  complex_query_percentage: [float, ...]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Configuration Loading Completeness

*For any* valid YAML configuration file containing all required fields, loading the configuration should successfully parse all fields and make them accessible in the resulting configuration object.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

### Property 2: Distribution Validation

*For any* configuration with distribution values (query_distribution or complexity_distribution), the system should validate that values sum to 1.0 within tolerance 0.001, and reject configurations outside this tolerance with descriptive error messages.

**Validates: Requirements 1.7**

### Property 3: Range Validation

*For any* configuration with accuracy values (routing_accuracy, delegation_accuracy), the system should validate that values are between 0.0 and 1.0 inclusive, and reject values outside this range with descriptive error messages.

**Validates: Requirements 1.8**

### Property 4: Token Estimation Includes System Prompts

*For any* query type, complexity, and model, the token estimator should always include the model's system prompt tokens in the total token estimate.

**Validates: Requirements 2.1**

### Property 5: Token Sampling Within Configured Ranges

*For any* query type and complexity, when estimating input tokens, the sampled value should fall within the configured minimum and maximum range for that query type and complexity.

**Validates: Requirements 2.2**

### Property 6: Output Token Multiplier Relationship

*For any* input token count and complexity level, the estimated output tokens should equal the input tokens multiplied by the complexity-specific output multiplier.

**Validates: Requirements 2.3**

### Property 7: Wrong Model Penalty Application

*For any* query where is_correct_model is False, the token estimate should be higher than the same query with is_correct_model True by the wrong_model_penalty multiplier.

**Validates: Requirements 2.4**

### Property 8: Delegation Cost Includes Both Components

*For any* query requiring delegation, the delegation cost should be greater than the tool execution cost alone, because it must include both tool cost and context reinjection cost.

**Validates: Requirements 2.6, 3.8**

### Property 9: Deterministic Estimation with Seed

*For any* query parameters and random seed, running token estimation twice with the same seed should produce identical token estimates.

**Validates: Requirements 2.7, 5.9**

### Property 10: Routing Cost Calculation

*For any* query, the routing cost should equal the classifier token usage multiplied by the classifier's per-token pricing (input + output).

**Validates: Requirements 3.1**

### Property 11: Lead Model Cost Calculation

*For any* query with known token counts, the lead model cost should equal (input_tokens × input_price) + (output_tokens × output_price) for the selected model.

**Validates: Requirements 3.5**

### Property 12: Misrouting Increases Cost

*For any* query that is misrouted, the total cost should be higher than if the same query were correctly routed, because misrouting requires correction by routing to the correct model.

**Validates: Requirements 3.4**

### Property 13: Routing Accuracy Probability

*For any* large sample of queries with routing_accuracy parameter, the proportion of queries correctly routed should converge to the routing_accuracy value within statistical tolerance.

**Validates: Requirements 3.3**

### Property 14: Medium Complexity Delegation Probability

*For any* large sample of MEDIUM complexity queries with delegation enabled, approximately 80% should have delegation_cost > 0.

**Validates: Requirements 3.6**

### Property 15: Complex Complexity Always Delegates

*For any* COMPLEX complexity query, the delegation_cost should be greater than 0, because COMPLEX queries always require delegation.

**Validates: Requirements 3.7**

### Property 16: Complex Second Delegation Probability

*For any* large sample of COMPLEX queries, approximately 30% should have num_delegations = 2, and those with second delegation should have higher total cost.

**Validates: Requirements 3.9**

### Property 17: Cost Breakdown Completeness

*For any* query cost calculation, the returned CostBreakdown should contain all required fields: routing_cost, lead_cost, delegation_cost, total_cost, query_type, complexity, was_routed_correctly, and num_delegations.

**Validates: Requirements 3.10**

### Property 18: Total Cost Equals Sum of Components

*For any* query cost calculation, the total_cost should equal routing_cost + lead_cost + delegation_cost.

**Validates: Requirements 3.10**

### Property 19: Simulation Executes Correct Number of Runs

*For any* simulation configuration with num_runs parameter, the simulation should execute exactly num_runs iterations.

**Validates: Requirements 5.1**

### Property 20: Simulation Executes Correct Number of Days

*For any* simulation configuration with num_days parameter, each run should simulate exactly num_days days.

**Validates: Requirements 5.2**

### Property 21: Simulation Generates Correct Number of Queries

*For any* simulation configuration with queries_per_day parameter, each simulated day should generate exactly queries_per_day query costs.

**Validates: Requirements 5.3**

### Property 22: Query Type Distribution Convergence

*For any* large simulation with query_distribution parameter, the proportion of each query type in the generated queries should converge to the specified distribution within statistical tolerance.

**Validates: Requirements 5.4**

### Property 23: Complexity Distribution Convergence

*For any* large simulation with complexity_distribution parameter, the proportion of each complexity level in the generated queries should converge to the specified distribution within statistical tolerance.

**Validates: Requirements 5.5**

### Property 24: Simulation Result Completeness

*For any* completed simulation, the SimulationResult should contain all required fields: mean_daily_cost, median_daily_cost, p95_daily_cost, p99_daily_cost, monthly_estimate, std_dev, routing_cost_share, lead_cost_share, delegation_cost_share, avg_delegations_per_query, and realized_routing_accuracy.

**Validates: Requirements 5.6, 5.7, 5.8**

### Property 25: Cost Breakdown Shares Sum to One

*For any* simulation result, the sum of routing_cost_share, lead_cost_share, and delegation_cost_share should equal 1.0 within tolerance 0.001.

**Validates: Requirements 5.7**

### Property 26: Monthly Estimate Calculation

*For any* simulation result, the monthly_estimate should equal mean_daily_cost × 30.

**Validates: Requirements 5.6**

### Property 27: Sensitivity Analysis Parameter Coverage

*For any* sensitivity analysis with specified parameters to vary, the results should include cost estimates for all configured values of each parameter.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

### Property 28: Sensitivity Delta Calculation

*For any* sensitivity analysis scenario, the percentage delta should equal ((scenario_cost - baseline_cost) / baseline_cost) × 100.

**Validates: Requirements 6.6**

### Property 29: CLI Parameter Override Application

*For any* CLI simulate command with parameter overrides, the simulation should use the override values instead of the profile's default values.

**Validates: Requirements 7.3**

### Property 30: CLI Output Completeness

*For any* CLI simulate command that completes successfully, the output should contain monthly estimate, daily averages, P95, P99, cost breakdown percentages, and metrics.

**Validates: Requirements 7.4**

### Property 31: CLI Compare Includes All Profiles

*For any* CLI compare command with multiple profile names, the output should include simulation results for each specified profile.

**Validates: Requirements 7.6**

### Property 32: CLI Output File Creation

*For any* CLI command with --output flag, the system should create a file at the specified path containing the command results.

**Validates: Requirements 7.9**

### Property 33: API Simulate Response Structure

*For any* valid POST /api/simulate request, the response should contain projection, breakdown, and metrics objects with all required fields.

**Validates: Requirements 8.1**

### Property 34: API Sensitivity Response Structure

*For any* valid POST /api/sensitivity request, the response should contain baseline and scenarios objects with cost estimates and deltas.

**Validates: Requirements 8.2**

### Property 35: API Validation Error Responses

*For any* invalid API request (missing required fields, invalid values), the API should return HTTP 400 status with a JSON error message describing the validation failure.

**Validates: Requirements 8.5**

### Property 36: API JSON Response Format

*For any* API endpoint response, the content should be valid JSON that can be parsed without errors.

**Validates: Requirements 8.6**

### Property 37: Config Editor Distribution Sum Validation

*For any* distribution slider values in the ConfigEditor, the validation function should return invalid when the sum is not within 0.001 of 1.0.

**Validates: Requirements 9.3**

### Property 38: Config Editor Profile Population

*For any* profile loaded in the ConfigEditor, all form fields should be populated with the corresponding profile values.

**Validates: Requirements 9.6**

### Property 39: Config Editor Validation State

*For any* invalid configuration in the ConfigEditor, the validation function should return false and the run button should be disabled.

**Validates: Requirements 9.7**

### Property 40: Monetary Value Formatting

*For any* monetary value displayed in the web interface, the formatted string should contain exactly 2 decimal places.

**Validates: Requirements 11.4**

### Property 41: Percentage Value Formatting

*For any* percentage value displayed in the web interface, the formatted string should contain exactly 1 decimal place.

**Validates: Requirements 11.5**

### Property 42: MAPE Calculation Correctness

*For any* set of predicted and actual values, the calculated MAPE should equal the mean of absolute percentage errors: mean(|actual - predicted| / actual × 100).

**Validates: Requirements 13.5**

### Property 43: Profile Save and Load Round Trip

*For any* valid UsageProfile, saving it to a file and then loading it back should produce an equivalent profile with all fields matching.

**Validates: Requirements 18.1, 18.2**

### Property 44: Built-in Profile Protection

*For any* built-in profile name (conservative, baseline, optimistic), attempting to delete it should raise an error and the profile file should remain unchanged.

**Validates: Requirements 18.5**

### Property 45: Profile Overwrite Protection

*For any* existing profile, attempting to save a profile with the same name without overwrite=True should raise an error and the original profile should remain unchanged.

**Validates: Requirements 18.6**

### Property 46: Profile Validation Before Save

*For any* invalid profile (distributions not summing to 1.0, invalid ranges), attempting to save it should raise a validation error with descriptive messages.

**Validates: Requirements 18.7**

### Property 47: Profile List Completeness

*For any* state of the profiles directory, listing profiles should return all .yaml files in the directory (excluding the .yaml extension from names).

**Validates: Requirements 18.1, 18.2**

## Error Handling

### Configuration Errors

**Invalid YAML Syntax:**
- Catch PyYAML parsing exceptions
- Return error with line number and description
- Example: "Invalid YAML syntax at line 15: unexpected character"

**Missing Required Fields:**
- Validate all required fields present after parsing
- Return error listing missing fields
- Example: "Missing required fields in profile: routing_accuracy, queries_per_day"

**Invalid Value Ranges:**
- Validate numeric ranges (0.0-1.0 for accuracies, > 0 for volumes)
- Return error with field name and valid range
- Example: "routing_accuracy must be between 0.0 and 1.0, got 1.5"

**Distribution Sum Errors:**
- Validate distributions sum to 1.0 ± 0.001
- Return error with actual sum
- Example: "query_distribution values sum to 1.05, must equal 1.0"

### Runtime Errors

**File Not Found:**
- Catch file I/O exceptions when loading configs
- Return error with file path
- Example: "Profile file not found: config/profiles/missing.yaml"

**Profile Management Errors:**
- Attempting to delete built-in profile
  - Example: "Cannot delete built-in profile 'baseline'"
- Attempting to overwrite without confirmation
  - Example: "Profile 'my-profile' already exists. Use --force to overwrite"
- Attempting to save invalid profile
  - Example: "Invalid profile: query_distribution sums to 1.05, must equal 1.0"
- Attempting to delete non-existent profile
  - Example: "Profile 'missing' not found"

**Invalid Delegation:**
- Validate delegation paths against rules
- Raise exception if invalid delegation attempted
- Example: "Invalid delegation: Coder cannot delegate to Gemini"

**API Request Errors:**
- Validate request payloads against schemas
- Return 400 with validation errors
- Example: "Invalid request: profile.routing_accuracy is required"

**Numerical Errors:**
- Handle division by zero in percentage calculations
- Handle negative costs (should never occur, indicates bug)
- Log warnings for unexpected values

### Error Recovery

**Graceful Degradation:**
- If optional config missing, use defaults
- Log warnings for missing optional fields
- Continue execution when possible

**Validation Before Execution:**
- Validate all inputs before starting expensive simulations
- Fail fast with clear error messages
- Prevent partial execution that wastes resources

## Testing Strategy

### Dual Testing Approach

The system will use both unit tests and property-based tests to ensure comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs
- Together they provide comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness

### Property-Based Testing Configuration

**Library Selection:**
- Python: Use `hypothesis` library for property-based testing
- Minimum 100 iterations per property test (configurable via `@settings(max_examples=100)`)

**Test Tagging:**
Each property test must include a comment referencing the design property:
```python
@given(...)
def test_token_estimation_includes_system_prompts(...):
    """
    Feature: llm-cost-predictor, Property 4: Token Estimation Includes System Prompts
    
    For any query type, complexity, and model, the token estimator should 
    always include the model's system prompt tokens in the total token estimate.
    """
```

**Generator Strategy:**
- Create custom Hypothesis strategies for domain types (QueryType, Complexity, ModelName)
- Generate valid configurations with constrained distributions (sum to 1.0)
- Generate realistic token ranges and pricing values
- Use `assume()` to filter invalid combinations

### Unit Testing Focus

**Specific Examples:**
- Test baseline profile produces expected cost range
- Test specific delegation paths (Gemini→Coder, Coder→Grok, etc.)
- Test edge cases: zero queries, 100% routing accuracy, 0% routing accuracy

**Integration Points:**
- Test CLI commands produce expected output formats
- Test API endpoints return valid JSON schemas
- Test configuration file loading round-trips correctly

**Error Conditions:**
- Test invalid YAML syntax raises appropriate errors
- Test missing required fields raises appropriate errors
- Test invalid value ranges raises appropriate errors
- Test invalid API requests return 400 status

### Test Organization

```
tests/
├── unit/
│   ├── test_types.py              # Enum and dataclass tests
│   ├── test_token_estimator.py    # Specific token estimation examples
│   ├── test_cost_predictor.py     # Specific cost calculation examples
│   ├── test_simulator.py          # Specific simulation examples
│   ├── test_sensitivity.py        # Specific sensitivity examples
│   ├── test_cli.py                # CLI command tests
│   └── test_api.py                # API endpoint tests
├── property/
│   ├── test_config_properties.py  # Properties 1-3
│   ├── test_token_properties.py   # Properties 4-9
│   ├── test_cost_properties.py    # Properties 10-18
│   ├── test_simulation_properties.py  # Properties 19-26
│   ├── test_sensitivity_properties.py # Properties 27-28
│   ├── test_cli_properties.py     # Properties 29-32
│   ├── test_api_properties.py     # Properties 33-36
│   └── test_ui_properties.py      # Properties 37-42
├── integration/
│   ├── test_end_to_end.py         # Full workflow tests
│   └── test_config_loading.py     # Config file loading tests
└── validate_estimates.py          # Real API validation script
```

### Validation Against Real APIs

**Critical Calibration Test:**
The `tests/validate_estimates.py` script is essential for ensuring simulation accuracy:

```python
# Pseudo-code for validation script
def validate_estimates():
    # Load hand-crafted test queries
    test_queries = load_test_queries()  # 5-10 queries spanning all types/complexities
    
    errors = []
    for query in test_queries:
        # Get prediction
        predicted_tokens = token_estimator.estimate(query)
        
        # Execute against real API
        actual_tokens = execute_real_api(query)
        
        # Calculate error
        error_pct = abs(actual_tokens - predicted_tokens) / actual_tokens * 100
        errors.append(error_pct)
    
    # Calculate MAPE
    mape = mean(errors)
    
    # Report results
    print(f"MAPE: {mape:.1f}%")
    for query, error in zip(test_queries, errors):
        print(f"  {query.type}/{query.complexity}: {error:.1f}% error")
    
    # Fail if MAPE > 30%
    assert mape <= 30.0, f"MAPE {mape:.1f}% exceeds 30% threshold"
```

**Test Query Coverage:**
- VISUAL/SIMPLE: "What color is this logo?" (with image)
- VISUAL/MEDIUM: "Analyze this chart and summarize trends" (with chart image)
- VISUAL/COMPLEX: "Compare these 3 diagrams and create a report" (with 3 images)

- CODE/SIMPLE: "Write a function to reverse a string"
- CODE/MEDIUM: "Create a REST API endpoint with validation"
- CODE/COMPLEX: "Build a complete authentication system with tests"

- RESEARCH/SIMPLE: "What is the capital of France?"
- RESEARCH/MEDIUM: "Summarize recent developments in quantum computing"
- RESEARCH/COMPLEX: "Compare 5 cloud providers and recommend one for our use case"

**Calibration Process:**
1. Run validation script against real APIs
2. If MAPE > 30%, analyze which query types have highest errors
3. Adjust token_estimation.yaml multipliers and ranges
4. Re-run validation until MAPE < 30%
5. Document calibration date and MAPE in docs/calibration.md

### Performance Testing

**Simulation Performance:**
- Measure time for 100 runs × 30 days simulation
- Assert completion time < 10 seconds
- Profile if performance degrades

**API Response Time:**
- Measure API endpoint response times
- Assert typical simulation request < 2 seconds
- Use async processing if needed for long simulations

### Test Execution

**Running Tests:**
```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/unit/

# Run only property tests
uv run pytest tests/property/

# Run with coverage
uv run pytest --cov=backend tests/

# Run validation against real APIs (requires API keys)
uv run python tests/validate_estimates.py
```

**Continuous Integration:**
- Run unit and property tests on every commit
- Run integration tests on pull requests
- Run validation tests weekly (requires API keys in CI)
- Fail build if any tests fail or coverage < 80%
