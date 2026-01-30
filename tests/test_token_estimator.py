"""Tests for token estimation module.

Tests Properties 9, 10, and 11 from the design document.
"""

import pytest
from hypothesis import given, settings, strategies as st

from core.token_estimator import TokenEstimator
from core.types import QueryType, Complexity, TokenEstimate


# Test configuration matching config/token_estimation.json
TEST_CONFIG = {
    "base_tokens": {
        "visual": {"input": 1000, "output": 500},
        "code": {"input": 800, "output": 2000},
        "research": {"input": 500, "output": 1500}
    },
    "complexity_multipliers": {
        "simple": 0.5,
        "moderate": 1.0,
        "complex": 2.5
    },
    "delegation_overhead": 1.5
}


@pytest.fixture
def estimator():
    """Create a TokenEstimator with test configuration."""
    return TokenEstimator(TEST_CONFIG)


class TestTokenEstimationByQueryType:
    """
    Property 9: Token Estimation by Query Type
    
    For any two queries with different query types but same complexity 
    and delegation status, the token estimates SHALL differ according 
    to the base_tokens configuration.
    
    **Validates: Requirements 2.1**
    """

    def test_visual_query_tokens(self, estimator):
        """Visual queries should use visual base tokens."""
        result = estimator.estimate(QueryType.VISUAL, Complexity.MODERATE, False)
        # moderate multiplier = 1.0, so tokens = base tokens
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_code_query_tokens(self, estimator):
        """Code queries should use code base tokens."""
        result = estimator.estimate(QueryType.CODE, Complexity.MODERATE, False)
        assert result.input_tokens == 800
        assert result.output_tokens == 2000

    def test_research_query_tokens(self, estimator):
        """Research queries should use research base tokens."""
        result = estimator.estimate(QueryType.RESEARCH, Complexity.MODERATE, False)
        assert result.input_tokens == 500
        assert result.output_tokens == 1500

    def test_different_query_types_produce_different_tokens(self, estimator):
        """Different query types should produce different token estimates."""
        visual = estimator.estimate(QueryType.VISUAL, Complexity.MODERATE, False)
        code = estimator.estimate(QueryType.CODE, Complexity.MODERATE, False)
        research = estimator.estimate(QueryType.RESEARCH, Complexity.MODERATE, False)
        
        # All should be different
        assert visual.input_tokens != code.input_tokens or visual.output_tokens != code.output_tokens
        assert visual.input_tokens != research.input_tokens or visual.output_tokens != research.output_tokens
        assert code.input_tokens != research.input_tokens or code.output_tokens != research.output_tokens


class TestComplexityMultiplierApplication:
    """
    Property 10: Complexity Multiplier Application
    
    For any query type, tokens estimated for "complex" queries SHALL be 
    greater than "moderate" which SHALL be greater than "simple", by the 
    configured multiplier ratios.
    
    **Validates: Requirements 2.2**
    """

    @settings(max_examples=100)
    @given(query_type=st.sampled_from(list(QueryType)))
    def test_complexity_ordering(self, query_type: QueryType):
        """Complex > moderate > simple for all query types."""
        estimator = TokenEstimator(TEST_CONFIG)
        
        simple = estimator.estimate(query_type, Complexity.SIMPLE, False)
        moderate = estimator.estimate(query_type, Complexity.MODERATE, False)
        complex_ = estimator.estimate(query_type, Complexity.COMPLEX, False)
        
        # Complex should have more tokens than moderate
        assert complex_.input_tokens > moderate.input_tokens
        assert complex_.output_tokens > moderate.output_tokens
        
        # Moderate should have more tokens than simple
        assert moderate.input_tokens > simple.input_tokens
        assert moderate.output_tokens > simple.output_tokens

    @settings(max_examples=100)
    @given(query_type=st.sampled_from(list(QueryType)))
    def test_complexity_multiplier_ratios(self, query_type: QueryType):
        """Verify multiplier ratios are correctly applied."""
        estimator = TokenEstimator(TEST_CONFIG)
        
        simple = estimator.estimate(query_type, Complexity.SIMPLE, False)
        moderate = estimator.estimate(query_type, Complexity.MODERATE, False)
        complex_ = estimator.estimate(query_type, Complexity.COMPLEX, False)
        
        # simple = base * 0.5, moderate = base * 1.0, complex = base * 2.5
        # So moderate/simple = 2.0 and complex/moderate = 2.5
        assert moderate.input_tokens == simple.input_tokens * 2
        assert moderate.output_tokens == simple.output_tokens * 2
        assert complex_.input_tokens == int(moderate.input_tokens * 2.5)
        assert complex_.output_tokens == int(moderate.output_tokens * 2.5)


class TestDelegationTokenOverhead:
    """
    Property 11: Delegation Token Overhead
    
    For any query, if delegation occurs, the token estimate SHALL be 
    multiplied by the delegation_overhead factor compared to the same 
    query without delegation.
    
    **Validates: Requirements 2.3**
    """

    @settings(max_examples=100)
    @given(
        query_type=st.sampled_from(list(QueryType)),
        complexity=st.sampled_from(list(Complexity))
    )
    def test_delegation_increases_tokens(self, query_type: QueryType, complexity: Complexity):
        """Delegated queries should have more tokens than non-delegated."""
        estimator = TokenEstimator(TEST_CONFIG)
        
        non_delegated = estimator.estimate(query_type, complexity, False)
        delegated = estimator.estimate(query_type, complexity, True)
        
        assert delegated.input_tokens > non_delegated.input_tokens
        assert delegated.output_tokens > non_delegated.output_tokens

    @settings(max_examples=100)
    @given(
        query_type=st.sampled_from(list(QueryType)),
        complexity=st.sampled_from(list(Complexity))
    )
    def test_delegation_overhead_factor(self, query_type: QueryType, complexity: Complexity):
        """Delegation should apply the configured overhead factor (1.5x)."""
        estimator = TokenEstimator(TEST_CONFIG)
        
        non_delegated = estimator.estimate(query_type, complexity, False)
        delegated = estimator.estimate(query_type, complexity, True)
        
        # delegation_overhead = 1.5
        expected_input = int(non_delegated.input_tokens * 1.5)
        expected_output = int(non_delegated.output_tokens * 1.5)
        
        assert delegated.input_tokens == expected_input
        assert delegated.output_tokens == expected_output
