"""Core domain types for the Multi-Model LLM Cost Predictor."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List


class QueryType(Enum):
    """Classification of query types."""
    VISUAL = "visual"
    CODE = "code"
    RESEARCH = "research"


class Complexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ModelName(Enum):
    """Available models in the system."""
    GEMINI = "gemini"
    CODER = "coder"
    GROK = "grok"
    CLASSIFIER = "classifier"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a single query."""
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
    """Configuration defining query distributions and usage patterns."""
    name: str
    description: str
    query_distribution: Dict[str, float]  # {visual: 0.2, code: 0.3, research: 0.5}
    complexity_distribution: Dict[str, float]  # {simple: 0.6, medium: 0.3, complex: 0.1}
    routing_accuracy: float
    delegation_accuracy: float
    queries_per_day: int

    def validate(self) -> List[str]:
        """
        Validate profile data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate query distribution
        if not self.query_distribution:
            errors.append("query_distribution is required")
        else:
            query_sum = sum(self.query_distribution.values())
            if abs(query_sum - 1.0) > 0.001:
                errors.append(
                    f"query_distribution values sum to {query_sum:.3f}, must equal 1.0"
                )
            
            # Check for required keys
            required_query_types = {"visual", "code", "research"}
            missing_types = required_query_types - set(self.query_distribution.keys())
            if missing_types:
                errors.append(
                    f"query_distribution missing required types: {', '.join(missing_types)}"
                )
            
            # Check for negative values
            for key, value in self.query_distribution.items():
                if value < 0:
                    errors.append(f"query_distribution[{key}] must be non-negative, got {value}")
        
        # Validate complexity distribution
        if not self.complexity_distribution:
            errors.append("complexity_distribution is required")
        else:
            complexity_sum = sum(self.complexity_distribution.values())
            if abs(complexity_sum - 1.0) > 0.001:
                errors.append(
                    f"complexity_distribution values sum to {complexity_sum:.3f}, must equal 1.0"
                )
            
            # Check for required keys
            required_complexities = {"simple", "medium", "complex"}
            missing_complexities = required_complexities - set(self.complexity_distribution.keys())
            if missing_complexities:
                errors.append(
                    f"complexity_distribution missing required complexities: {', '.join(missing_complexities)}"
                )
            
            # Check for negative values
            for key, value in self.complexity_distribution.items():
                if value < 0:
                    errors.append(f"complexity_distribution[{key}] must be non-negative, got {value}")
        
        # Validate routing accuracy
        if not (0.0 <= self.routing_accuracy <= 1.0):
            errors.append(
                f"routing_accuracy must be between 0.0 and 1.0, got {self.routing_accuracy}"
            )
        
        # Validate delegation accuracy
        if not (0.0 <= self.delegation_accuracy <= 1.0):
            errors.append(
                f"delegation_accuracy must be between 0.0 and 1.0, got {self.delegation_accuracy}"
            )
        
        # Validate queries per day
        if self.queries_per_day <= 0:
            errors.append(
                f"queries_per_day must be greater than 0, got {self.queries_per_day}"
            )
        
        return errors


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""
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
