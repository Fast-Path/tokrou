"""Token estimation logic for different query types and models."""

import yaml
from typing import Dict, Tuple, Optional
import numpy as np

from .types import QueryType, Complexity, ModelName


class TokenEstimator:
    """Predicts token usage for queries based on type, complexity, and model."""
    
    def __init__(self, config: Dict):
        """
        Initialize with token estimation configuration.
        
        Args:
            config: Dictionary containing token estimation rules from token_estimation.yaml
        """
        self.config = config
        self.system_prompts = config['system_prompts']
        self.input_ranges = config['input_ranges']
        self.output_multipliers = config['output_multipliers']
        self.wrong_model_penalty = config['wrong_model_penalty']
    
    @classmethod
    def from_config_file(cls, config_path: str = 'config/token_estimation.yaml') -> 'TokenEstimator':
        """
        Create TokenEstimator from YAML configuration file.
        
        Args:
            config_path: Path to token estimation configuration file
            
        Returns:
            TokenEstimator instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def estimate_classifier_tokens(self, query_text: Optional[str] = None) -> int:
        """
        Estimate tokens for classifier routing decision.
        
        Args:
            query_text: Optional query text (not used in current implementation)
            
        Returns:
            Number of tokens for classifier invocation
        """
        # Classifier uses a fixed system prompt size
        # In practice, this would include the query text, but for simulation
        # we use the system prompt size as a baseline
        return self.system_prompts['classifier']
    
    def estimate_input_tokens(
        self,
        query_type: QueryType,
        complexity: Complexity,
        model: ModelName,
        rng: np.random.Generator
    ) -> int:
        """
        Sample input tokens from configured range for query type and complexity.
        
        Args:
            query_type: Type of query (VISUAL, CODE, RESEARCH)
            complexity: Complexity level (SIMPLE, MEDIUM, COMPLEX)
            model: Target model (used for system prompt)
            rng: Random number generator for sampling
            
        Returns:
            Sampled input token count including system prompt
        """
        # Get the range for this query type and complexity
        min_tokens, max_tokens = self.input_ranges[query_type.value][complexity.value]
        
        # Sample uniformly from the range
        sampled_tokens = rng.integers(min_tokens, max_tokens + 1)
        
        # Add system prompt tokens for the target model
        system_prompt_tokens = self.system_prompts[model.value]
        
        return sampled_tokens + system_prompt_tokens
    
    def estimate_output_tokens(
        self,
        input_tokens: int,
        complexity: Complexity
    ) -> int:
        """
        Calculate output tokens using complexity multiplier.
        
        Args:
            input_tokens: Number of input tokens (excluding system prompt)
            complexity: Complexity level determining output multiplier
            
        Returns:
            Estimated output token count
        """
        multiplier = self.output_multipliers[complexity.value]
        return int(input_tokens * multiplier)
    
    def estimate_total_tokens(
        self,
        query_type: QueryType,
        complexity: Complexity,
        model: ModelName,
        is_correct_model: bool,
        rng: np.random.Generator
    ) -> Tuple[int, int]:
        """
        Estimate total input and output tokens for a query.
        
        Args:
            query_type: Type of query (VISUAL, CODE, RESEARCH)
            complexity: Complexity level (SIMPLE, MEDIUM, COMPLEX)
            model: Target model for processing
            is_correct_model: Whether this is the optimal model for the query type
            rng: Random number generator for sampling
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        # Get base input tokens (includes system prompt)
        input_tokens = self.estimate_input_tokens(query_type, complexity, model, rng)
        
        # Calculate base output tokens (based on input minus system prompt)
        base_input_for_output = input_tokens - self.system_prompts[model.value]
        output_tokens = self.estimate_output_tokens(base_input_for_output, complexity)
        
        # Apply wrong model penalty if needed
        if not is_correct_model:
            input_tokens = int(input_tokens * self.wrong_model_penalty)
            output_tokens = int(output_tokens * self.wrong_model_penalty)
        
        return input_tokens, output_tokens
    
    def estimate_delegation_tokens(
        self,
        query_type: QueryType,
        complexity: Complexity,
        tool_model: ModelName,
        original_input_tokens: int,
        rng: np.random.Generator
    ) -> Tuple[int, int, int, int]:
        """
        Estimate tokens for delegation including tool execution and context reinjection.
        
        Args:
            query_type: Type of query being delegated
            complexity: Complexity level
            tool_model: Model being used as a tool
            original_input_tokens: Input tokens from the original query
            rng: Random number generator for sampling
            
        Returns:
            Tuple of (tool_input_tokens, tool_output_tokens, context_input_tokens, context_output_tokens)
        """
        # Tool execution tokens
        tool_input_tokens, tool_output_tokens = self.estimate_total_tokens(
            query_type, complexity, tool_model, True, rng
        )
        
        # Context reinjection: original query + tool output
        # The lead model needs to process the original input plus the tool's output
        context_input_tokens = original_input_tokens + tool_output_tokens
        
        # Context output based on the combined input
        base_context_input = context_input_tokens - self.system_prompts[tool_model.value]
        context_output_tokens = self.estimate_output_tokens(base_context_input, complexity)
        
        return tool_input_tokens, tool_output_tokens, context_input_tokens, context_output_tokens