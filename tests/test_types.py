"""Property-based tests for core type definitions.

Tests the round-trip serialization property for dataclasses.
"""

import pytest
from hypothesis import given, settings, strategies as st

from core.types import (
    SimulationResult,
    CostBreakdown,
)


# Strategies for generating valid test data
cost_breakdown_strategy = st.builds(
    CostBreakdown,
    routing_cost=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    lead_model_cost=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    delegation_cost=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    retry_cost=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    total_cost=st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
)

simulation_result_strategy = st.builds(
    SimulationResult,
    profile_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
    iterations=st.integers(min_value=1, max_value=100000),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    daily_costs=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
    cost_breakdown=cost_breakdown_strategy,
    mean_cost=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    p50_cost=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    p95_cost=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    p99_cost=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    total_revenue=st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False),
    profit_margin=st.floats(min_value=-10, max_value=1, allow_nan=False, allow_infinity=False),
    rejection_rate=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    avg_latency_ms=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    cache_savings=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=100)
@given(result=simulation_result_strategy)
def test_simulation_result_round_trip_serialization(result: SimulationResult):
    """
    Property 13: Dataclass Round-Trip Serialization
    
    For any valid SimulationResult instance, serializing to JSON then 
    deserializing SHALL produce an object equivalent to the original.
    
    **Validates: Requirements 9.4**
    """
    # Serialize to JSON
    json_str = result.to_json()
    
    # Deserialize back
    restored = SimulationResult.from_json(json_str)
    
    # Verify equivalence
    assert restored.profile_name == result.profile_name
    assert restored.iterations == result.iterations
    assert restored.seed == result.seed
    assert restored.daily_costs == result.daily_costs
    assert restored.mean_cost == result.mean_cost
    assert restored.p50_cost == result.p50_cost
    assert restored.p95_cost == result.p95_cost
    assert restored.p99_cost == result.p99_cost
    assert restored.total_revenue == result.total_revenue
    assert restored.profit_margin == result.profit_margin
    assert restored.rejection_rate == result.rejection_rate
    assert restored.avg_latency_ms == result.avg_latency_ms
    assert restored.cache_savings == result.cache_savings
    
    # Verify cost breakdown
    assert restored.cost_breakdown.routing_cost == result.cost_breakdown.routing_cost
    assert restored.cost_breakdown.lead_model_cost == result.cost_breakdown.lead_model_cost
    assert restored.cost_breakdown.delegation_cost == result.cost_breakdown.delegation_cost
    assert restored.cost_breakdown.retry_cost == result.cost_breakdown.retry_cost
    assert restored.cost_breakdown.total_cost == result.cost_breakdown.total_cost
