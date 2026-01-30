"""Unit tests for configuration loader.

Tests loading valid and invalid configuration files.
"""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from core.config import ConfigLoader
from core.types import QueryType, Complexity, UserTier, ModelPricing, TierConfig, Profile


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with valid test files."""
    temp_dir = Path(tempfile.mkdtemp())
    profiles_dir = temp_dir / "profiles"
    profiles_dir.mkdir()
    
    # Create pricing.json
    pricing_data = {
        "models": {
            "classifier": {
                "input_price_per_1k": 0.001,
                "output_price_per_1k": 0.002,
                "latency_ms_avg": 100
            },
            "gemini": {
                "input_price_per_1k": 0.01,
                "output_price_per_1k": 0.03,
                "cached_input_price_per_1k": 0.005,
                "latency_ms_avg": 500,
                "availability_rate": 0.995
            }
        }
    }
    (temp_dir / "pricing.json").write_text(json.dumps(pricing_data))
    
    # Create default profile
    profile_data = {
        "name": "test-profile",
        "daily_volume": 1000,
        "query_mix": {"visual": 0.3, "code": 0.5, "research": 0.2},
        "complexity_distribution": {"simple": 0.5, "moderate": 0.3, "complex": 0.2},
        "routing_accuracy": 0.9,
        "delegation_probability": 0.3,
        "cache_hit_rate": 0.2,
        "error_rate": 0.02,
        "tier_distribution": {"free": 0.7, "paid": 0.3},
        "user_counts": {"free": 500, "paid": 200}
    }
    (profiles_dir / "test-profile.json").write_text(json.dumps(profile_data))
    
    # Create token_estimation.json
    token_data = {
        "base_tokens": {
            "visual": {"input": 1000, "output": 500},
            "code": {"input": 800, "output": 1200},
            "research": {"input": 600, "output": 2000}
        },
        "complexity_multipliers": {"simple": 1.0, "moderate": 1.5, "complex": 2.5},
        "delegation_overhead": 1.5
    }
    (temp_dir / "token_estimation.json").write_text(json.dumps(token_data))
    
    # Create routing.json
    routing_data = {
        "fallback_chains": {
            "gemini": ["coder", "grok"],
            "coder": ["gemini"],
            "grok": ["coder"]
        }
    }
    (temp_dir / "routing.json").write_text(json.dumps(routing_data))
    
    # Create tiers.json
    tiers_data = {
        "tiers": {
            "free": {
                "daily_query_limit": 50,
                "allowed_complexity": ["simple", "moderate"],
                "subscription_fee_monthly": 0,
                "growth_rate_monthly": 0.1
            },
            "paid": {
                "daily_query_limit": 500,
                "allowed_complexity": ["simple", "moderate", "complex"],
                "subscription_fee_monthly": 29.99,
                "growth_rate_monthly": 0.05
            }
        }
    }
    (temp_dir / "tiers.json").write_text(json.dumps(tiers_data))
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestConfigLoaderValid:
    """Tests for loading valid configuration files."""

    def test_load_pricing(self, temp_config_dir):
        """Test loading valid pricing configuration."""
        loader = ConfigLoader(temp_config_dir)
        pricing = loader.load_pricing()
        
        assert "classifier" in pricing
        assert "gemini" in pricing
        assert isinstance(pricing["classifier"], ModelPricing)
        assert pricing["classifier"].input_price_per_1k == 0.001
        assert pricing["gemini"].cached_input_price_per_1k == 0.005
        assert pricing["gemini"].availability_rate == 0.995
        # Check default cached price (50% of input)
        assert pricing["classifier"].cached_input_price_per_1k == 0.0005

    def test_load_profile(self, temp_config_dir):
        """Test loading valid profile configuration."""
        loader = ConfigLoader(temp_config_dir)
        profile = loader.load_profile("test-profile")
        
        assert isinstance(profile, Profile)
        assert profile.name == "test-profile"
        assert profile.daily_volume == 1000
        assert profile.query_mix[QueryType.CODE] == 0.5
        assert profile.complexity_distribution[Complexity.SIMPLE] == 0.5
        assert profile.routing_accuracy == 0.9
        assert profile.tier_distribution[UserTier.FREE] == 0.7

    def test_load_token_estimation(self, temp_config_dir):
        """Test loading token estimation configuration."""
        loader = ConfigLoader(temp_config_dir)
        token_config = loader.load_token_estimation()
        
        assert "base_tokens" in token_config
        assert "complexity_multipliers" in token_config
        assert token_config["base_tokens"]["code"]["output"] == 1200

    def test_load_routing(self, temp_config_dir):
        """Test loading routing configuration."""
        loader = ConfigLoader(temp_config_dir)
        routing = loader.load_routing()
        
        assert "fallback_chains" in routing
        assert "gemini" in routing["fallback_chains"]

    def test_load_tiers(self, temp_config_dir):
        """Test loading tier configuration."""
        loader = ConfigLoader(temp_config_dir)
        tiers = loader.load_tiers()
        
        assert UserTier.FREE in tiers
        assert UserTier.PAID in tiers
        assert isinstance(tiers[UserTier.FREE], TierConfig)
        assert tiers[UserTier.FREE].daily_query_limit == 50
        assert tiers[UserTier.PAID].subscription_fee_monthly == 29.99
        assert Complexity.COMPLEX not in tiers[UserTier.FREE].allowed_complexity


class TestConfigLoaderInvalid:
    """Tests for handling invalid configuration files."""

    def test_missing_pricing_file(self, temp_config_dir):
        """Test error when pricing.json is missing."""
        (temp_config_dir / "pricing.json").unlink()
        loader = ConfigLoader(temp_config_dir)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_pricing()
        assert "pricing.json" in str(exc_info.value)

    def test_missing_profile(self, temp_config_dir):
        """Test error when profile doesn't exist."""
        loader = ConfigLoader(temp_config_dir)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_profile("nonexistent")
        assert "nonexistent.json" in str(exc_info.value)

    def test_invalid_json(self, temp_config_dir):
        """Test error when JSON is malformed."""
        (temp_config_dir / "pricing.json").write_text("{invalid json}")
        loader = ConfigLoader(temp_config_dir)
        
        with pytest.raises(ValueError) as exc_info:
            loader.load_pricing()
        assert "Invalid JSON" in str(exc_info.value)

    def test_missing_config_dir(self):
        """Test error when config directory doesn't exist."""
        loader = ConfigLoader(Path("/nonexistent/path"))
        
        with pytest.raises(FileNotFoundError):
            loader.load_pricing()
