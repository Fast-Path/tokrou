"""Token estimation module for LLM Cost Predictor.

Estimates token counts based on query characteristics including
query type, complexity, and delegation status.
"""

from .types import QueryType, Complexity, TokenEstimate


class TokenEstimator:
    """Estimates input and output tokens for queries based on configuration."""

    def __init__(self, estimation_config: dict):
        """Initialize the token estimator with configuration.
        
        Args:
            estimation_config: Dictionary containing:
                - base_tokens: Dict mapping query type to {input, output} token counts
                - complexity_multipliers: Dict mapping complexity to multiplier values
                - delegation_overhead: Float multiplier for delegated queries
        """
        self.config = estimation_config

    def estimate(self, query_type: QueryType, complexity: Complexity,
                 delegated: bool) -> TokenEstimate:
        """Estimate input and output tokens for a query.
        
        Args:
            query_type: The type of query (visual, code, research)
            complexity: The complexity level (simple, moderate, complex)
            delegated: Whether the query was delegated to another model
            
        Returns:
            TokenEstimate with input_tokens and output_tokens
        """
        base = self.config["base_tokens"].get(query_type.value, {})
        multiplier = self.config["complexity_multipliers"].get(complexity.value, 1.0)

        input_tokens = int(base.get("input", 500) * multiplier)
        output_tokens = int(base.get("output", 1000) * multiplier)

        if delegated:
            delegation_overhead = self.config.get("delegation_overhead", 1.5)
            input_tokens = int(input_tokens * delegation_overhead)
            output_tokens = int(output_tokens * delegation_overhead)

        return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)
