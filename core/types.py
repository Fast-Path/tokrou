"""Core type definitions for LLM Cost Predictor.

This module defines all data structures using dataclasses and enums
for the Monte Carlo simulation engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


class QueryType(Enum):
    """Category of query being processed."""
    VISUAL = "visual"
    CODE = "code"
    RESEARCH = "research"


class Complexity(Enum):
    """Query difficulty level."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class UserTier(Enum):
    """User subscription tier."""
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"


@dataclass
class ModelPricing:
    """Pricing configuration for a single model."""
    name: str
    input_price_per_1k: float  # USD per 1000 tokens
    output_price_per_1k: float
    cached_input_price_per_1k: float
    latency_ms_avg: float
    availability_rate: float = 0.999


@dataclass
class TokenEstimate:
    """Estimated token counts for a query."""
    input_tokens: int
    output_tokens: int


@dataclass
class QuerySimulation:
    """Result of simulating a single query."""
    query_type: QueryType
    complexity: Complexity
    user_tier: UserTier
    routed_to: str
    correct_route: bool
    delegated: bool
    delegation_chain: list[str]
    tokens: TokenEstimate
    cache_hit: bool
    error_occurred: bool
    retry_count: int
    cost_usd: float
    latency_ms: float


@dataclass
class CostBreakdown:
    """Breakdown of costs by category."""
    routing_cost: float
    lead_model_cost: float
    delegation_cost: float
    retry_cost: float
    total_cost: float


@dataclass
class TierConfig:
    """Configuration for a user tier."""
    name: UserTier
    daily_query_limit: int | None
    allowed_complexity: list[Complexity]
    subscription_fee_monthly: float
    growth_rate_monthly: float


@dataclass
class Profile:
    """Usage profile defining simulation parameters."""
    name: str
    daily_volume: int
    query_mix: dict[QueryType, float]  # Probabilities summing to 1.0
    complexity_distribution: dict[Complexity, float]
    routing_accuracy: float
    delegation_probability: float
    cache_hit_rate: float
    error_rate: float
    tier_distribution: dict[UserTier, float]
    user_counts: dict[UserTier, int]


@dataclass
class SimulationResult:
    """Complete results from a Monte Carlo simulation run."""
    profile_name: str
    iterations: int
    seed: int
    daily_costs: list[float]
    cost_breakdown: CostBreakdown
    mean_cost: float
    p50_cost: float
    p95_cost: float
    p99_cost: float
    total_revenue: float
    profit_margin: float
    rejection_rate: float
    avg_latency_ms: float
    cache_savings: float

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "profile_name": self.profile_name,
            "iterations": self.iterations,
            "seed": self.seed,
            "daily_costs": self.daily_costs,
            "cost_breakdown": {
                "routing_cost": self.cost_breakdown.routing_cost,
                "lead_model_cost": self.cost_breakdown.lead_model_cost,
                "delegation_cost": self.cost_breakdown.delegation_cost,
                "retry_cost": self.cost_breakdown.retry_cost,
                "total_cost": self.cost_breakdown.total_cost,
            },
            "mean_cost": self.mean_cost,
            "p50_cost": self.p50_cost,
            "p95_cost": self.p95_cost,
            "p99_cost": self.p99_cost,
            "total_revenue": self.total_revenue,
            "profit_margin": self.profit_margin,
            "rejection_rate": self.rejection_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "cache_savings": self.cache_savings,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationResult":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        cost_breakdown = CostBreakdown(
            routing_cost=data["cost_breakdown"]["routing_cost"],
            lead_model_cost=data["cost_breakdown"]["lead_model_cost"],
            delegation_cost=data["cost_breakdown"]["delegation_cost"],
            retry_cost=data["cost_breakdown"]["retry_cost"],
            total_cost=data["cost_breakdown"]["total_cost"],
        )
        return cls(
            profile_name=data["profile_name"],
            iterations=data["iterations"],
            seed=data["seed"],
            daily_costs=data["daily_costs"],
            cost_breakdown=cost_breakdown,
            mean_cost=data["mean_cost"],
            p50_cost=data["p50_cost"],
            p95_cost=data["p95_cost"],
            p99_cost=data["p99_cost"],
            total_revenue=data["total_revenue"],
            profit_margin=data["profit_margin"],
            rejection_rate=data["rejection_rate"],
            avg_latency_ms=data["avg_latency_ms"],
            cache_savings=data["cache_savings"],
        )
