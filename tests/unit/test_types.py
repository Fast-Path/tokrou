"""Unit tests for core domain types."""

import pytest
from backend.core import (
    QueryType,
    Complexity,
    ModelName,
    CostBreakdown,
    UsageProfile,
    SimulationResult,
)


class TestEnums:
    """Test enum definitions."""

    def test_query_type_values(self):
        """Test QueryType enum has correct values."""
        assert QueryType.VISUAL.value == "visual"
        assert QueryType.CODE.value == "code"
        assert QueryType.RESEARCH.value == "research"

    def test_complexity_values(self):
        """Test Complexity enum has correct values."""
        assert Complexity.SIMPLE.value == "simple"
        assert Complexity.MEDIUM.value == "medium"
        assert Complexity.COMPLEX.value == "complex"

    def test_model_name_values(self):
        """Test ModelName enum has correct values."""
        assert ModelName.GEMINI.value == "gemini"
        assert ModelName.CODER.value == "coder"
        assert ModelName.GROK.value == "grok"
        assert ModelName.CLASSIFIER.value == "classifier"


class TestCostBreakdown:
    """Test CostBreakdown dataclass."""

    def test_cost_breakdown_creation(self):
        """Test creating a CostBreakdown instance."""
        breakdown = CostBreakdown(
            routing_cost=0.001,
            lead_cost=0.05,
            delegation_cost=0.02,
            total_cost=0.071,
            query_type=QueryType.CODE,
            complexity=Complexity.MEDIUM,
            was_routed_correctly=True,
            num_delegations=1,
        )
        
        assert breakdown.routing_cost == 0.001
        assert breakdown.lead_cost == 0.05
        assert breakdown.delegation_cost == 0.02
        assert breakdown.total_cost == 0.071
        assert breakdown.query_type == QueryType.CODE
        assert breakdown.complexity == Complexity.MEDIUM
        assert breakdown.was_routed_correctly is True
        assert breakdown.num_delegations == 1


class TestUsageProfile:
    """Test UsageProfile dataclass and validation."""

    def test_usage_profile_creation(self):
        """Test creating a valid UsageProfile instance."""
        profile = UsageProfile(
            name="test-profile",
            description="Test profile",
            query_distribution={"visual": 0.2, "code": 0.3, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=1000,
        )
        
        assert profile.name == "test-profile"
        assert profile.description == "Test profile"
        assert profile.routing_accuracy == 0.9
        assert profile.delegation_accuracy == 0.95
        assert profile.queries_per_day == 1000

    def test_usage_profile_validation_valid(self):
        """Test validation passes for valid profile."""
        profile = UsageProfile(
            name="valid",
            description="Valid profile",
            query_distribution={"visual": 0.2, "code": 0.3, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=1000,
        )
        
        errors = profile.validate()
        assert errors == []

    def test_usage_profile_validation_distribution_sum_invalid(self):
        """Test validation fails when distributions don't sum to 1.0."""
        profile = UsageProfile(
            name="invalid",
            description="Invalid profile",
            query_distribution={"visual": 0.3, "code": 0.3, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.2},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=1000,
        )
        
        errors = profile.validate()
        assert len(errors) == 2
        assert any("query_distribution" in error and "1.100" in error for error in errors)
        assert any("complexity_distribution" in error and "1.100" in error for error in errors)

    def test_usage_profile_validation_accuracy_out_of_range(self):
        """Test validation fails when accuracy values are out of range."""
        profile = UsageProfile(
            name="invalid",
            description="Invalid profile",
            query_distribution={"visual": 0.2, "code": 0.3, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=1.5,
            delegation_accuracy=-0.1,
            queries_per_day=1000,
        )
        
        errors = profile.validate()
        assert len(errors) == 2
        assert any("routing_accuracy" in error and "1.5" in error for error in errors)
        assert any("delegation_accuracy" in error and "-0.1" in error for error in errors)

    def test_usage_profile_validation_queries_per_day_invalid(self):
        """Test validation fails when queries_per_day is not positive."""
        profile = UsageProfile(
            name="invalid",
            description="Invalid profile",
            query_distribution={"visual": 0.2, "code": 0.3, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=0,
        )
        
        errors = profile.validate()
        assert len(errors) == 1
        assert "queries_per_day" in errors[0]

    def test_usage_profile_validation_missing_query_types(self):
        """Test validation fails when query distribution is missing required types."""
        profile = UsageProfile(
            name="invalid",
            description="Invalid profile",
            query_distribution={"visual": 0.5, "code": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=1000,
        )
        
        errors = profile.validate()
        assert len(errors) == 1
        assert "research" in errors[0]

    def test_usage_profile_validation_negative_distribution_values(self):
        """Test validation fails when distribution values are negative."""
        profile = UsageProfile(
            name="invalid",
            description="Invalid profile",
            query_distribution={"visual": -0.1, "code": 0.6, "research": 0.5},
            complexity_distribution={"simple": 0.6, "medium": 0.3, "complex": 0.1},
            routing_accuracy=0.9,
            delegation_accuracy=0.95,
            queries_per_day=1000,
        )
        
        errors = profile.validate()
        assert any("query_distribution[visual]" in error and "negative" in error for error in errors)


class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_simulation_result_creation(self):
        """Test creating a SimulationResult instance."""
        result = SimulationResult(
            mean_daily_cost=10.5,
            median_daily_cost=10.2,
            p95_daily_cost=15.3,
            p99_daily_cost=18.7,
            monthly_estimate=315.0,
            std_dev=2.3,
            routing_cost_share=0.05,
            lead_cost_share=0.70,
            delegation_cost_share=0.25,
            avg_delegations_per_query=0.4,
            realized_routing_accuracy=0.89,
        )
        
        assert result.mean_daily_cost == 10.5
        assert result.median_daily_cost == 10.2
        assert result.p95_daily_cost == 15.3
        assert result.p99_daily_cost == 18.7
        assert result.monthly_estimate == 315.0
        assert result.std_dev == 2.3
        assert result.routing_cost_share == 0.05
        assert result.lead_cost_share == 0.70
        assert result.delegation_cost_share == 0.25
        assert result.avg_delegations_per_query == 0.4
        assert result.realized_routing_accuracy == 0.89
