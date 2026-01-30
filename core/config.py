"""Configuration loader for LLM Cost Predictor.

Handles loading and validating JSON configuration files for pricing,
profiles, token estimation, routing, and tier configurations.
"""

import json
from pathlib import Path
from .types import Profile, ModelPricing, TierConfig, QueryType, Complexity, UserTier


class ConfigLoader:
    """Loads and validates configuration files for the cost predictor."""

    def __init__(self, config_dir: Path):
        """Initialize the config loader with a configuration directory.
        
        Args:
            config_dir: Path to the directory containing configuration files.
        """
        self.config_dir = Path(config_dir)

    def load_pricing(self) -> dict[str, ModelPricing]:
        """Load model pricing configuration.
        
        Returns:
            Dictionary mapping model names to ModelPricing objects.
            
        Raises:
            FileNotFoundError: If pricing.json doesn't exist.
            ValueError: If the JSON is invalid.
        """
        path = self.config_dir / "pricing.json"
        data = self._load_json(path)
        return {
            name: ModelPricing(
                name=name,
                input_price_per_1k=cfg["input_price_per_1k"],
                output_price_per_1k=cfg["output_price_per_1k"],
                cached_input_price_per_1k=cfg.get(
                    "cached_input_price_per_1k", cfg["input_price_per_1k"] * 0.5
                ),
                latency_ms_avg=cfg.get("latency_ms_avg", 1000),
                availability_rate=cfg.get("availability_rate", 0.999),
            )
            for name, cfg in data["models"].items()
        }

    def load_profile(self, name: str) -> Profile:
        """Load a usage profile.
        
        Args:
            name: Name of the profile (without .json extension).
            
        Returns:
            Profile object with the loaded configuration.
            
        Raises:
            FileNotFoundError: If the profile doesn't exist.
            ValueError: If the JSON is invalid.
        """
        path = self.config_dir / "profiles" / f"{name}.json"
        data = self._load_json(path)
        return Profile(
            name=data["name"],
            daily_volume=data["daily_volume"],
            query_mix={QueryType(k): v for k, v in data["query_mix"].items()},
            complexity_distribution={
                Complexity(k): v for k, v in data["complexity_distribution"].items()
            },
            routing_accuracy=data["routing_accuracy"],
            delegation_probability=data["delegation_probability"],
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
            error_rate=data.get("error_rate", 0.01),
            tier_distribution={
                UserTier(k): v
                for k, v in data.get(
                    "tier_distribution", {"free": 0.8, "paid": 0.2}
                ).items()
            },
            user_counts={
                UserTier(k): v
                for k, v in data.get(
                    "user_counts", {"free": 1000, "paid": 100}
                ).items()
            },
        )

    def load_token_estimation(self) -> dict:
        """Load token estimation rules.
        
        Returns:
            Dictionary containing token estimation configuration.
            
        Raises:
            FileNotFoundError: If token_estimation.json doesn't exist.
            ValueError: If the JSON is invalid.
        """
        path = self.config_dir / "token_estimation.json"
        return self._load_json(path)

    def load_routing(self) -> dict:
        """Load routing configuration.
        
        Returns:
            Dictionary containing routing configuration.
            
        Raises:
            FileNotFoundError: If routing.json doesn't exist.
            ValueError: If the JSON is invalid.
        """
        path = self.config_dir / "routing.json"
        return self._load_json(path)

    def load_tiers(self) -> dict[UserTier, TierConfig]:
        """Load tier configurations.
        
        Returns:
            Dictionary mapping UserTier to TierConfig objects.
            
        Raises:
            FileNotFoundError: If tiers.json doesn't exist.
            ValueError: If the JSON is invalid.
        """
        path = self.config_dir / "tiers.json"
        data = self._load_json(path)
        return {
            UserTier(name): TierConfig(
                name=UserTier(name),
                daily_query_limit=cfg.get("daily_query_limit"),
                allowed_complexity=[
                    Complexity(c)
                    for c in cfg.get(
                        "allowed_complexity", ["simple", "moderate", "complex"]
                    )
                ],
                subscription_fee_monthly=cfg.get("subscription_fee_monthly", 0),
                growth_rate_monthly=cfg.get("growth_rate_monthly", 0.0),
            )
            for name, cfg in data["tiers"].items()
        }

    def _load_json(self, path: Path) -> dict:
        """Load and parse a JSON file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            Parsed JSON data as a dictionary.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
