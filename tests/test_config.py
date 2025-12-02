"""Tests for configuration system."""

from proactive_hcdt.ai_providers import DummyAIProvider
from proactive_hcdt.config.settings import (
    AIProviderConfig,
    AIProviderType,
    FrameworkConfig,
    create_config,
)


class TestAIProviderConfig:
    """Tests for AIProviderConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AIProviderConfig()

        assert config.provider_type == AIProviderType.DUMMY
        assert config.model_name == "dummy-model-v1"

    def test_gemini_default_model(self):
        """Test Gemini default model name."""
        config = AIProviderConfig(provider_type=AIProviderType.GEMINI)

        assert config.model_name == "gemini-1.5-pro"

    def test_openai_default_model(self):
        """Test OpenAI default model name."""
        config = AIProviderConfig(provider_type=AIProviderType.OPENAI)

        assert config.model_name == "gpt-4-turbo-preview"

    def test_anthropic_default_model(self):
        """Test Anthropic default model name."""
        config = AIProviderConfig(provider_type=AIProviderType.ANTHROPIC)

        assert config.model_name == "claude-3-5-sonnet-20241022"

    def test_custom_model_name(self):
        """Test custom model name is preserved."""
        config = AIProviderConfig(
            provider_type=AIProviderType.OPENAI,
            model_name="gpt-4o"
        )

        assert config.model_name == "gpt-4o"

    def test_create_dummy_provider(self):
        """Test creating dummy provider."""
        config = AIProviderConfig(provider_type=AIProviderType.DUMMY)
        provider = config.create_provider()

        assert isinstance(provider, DummyAIProvider)


class TestFrameworkConfig:
    """Tests for FrameworkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FrameworkConfig()

        assert config.robot_name == "ProactiveBot"
        assert config.max_tool_iterations == 10
        assert config.enable_proactive_scanning is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "ai_provider": {
                "provider_type": "dummy",
                "model_name": "test-model"
            },
            "robot_name": "TestBot",
            "debug_mode": True
        }

        config = FrameworkConfig.from_dict(config_dict)

        assert config.robot_name == "TestBot"
        assert config.debug_mode is True
        assert config.ai_provider.model_name == "test-model"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = FrameworkConfig(
            robot_name="MyBot",
            debug_mode=True
        )

        config_dict = config.to_dict()

        assert config_dict["robot_name"] == "MyBot"
        assert config_dict["debug_mode"] is True
        assert "ai_provider" in config_dict

    def test_safety_settings(self):
        """Test safety settings."""
        config = FrameworkConfig(
            require_confirmation_for_movement=True,
            max_movement_distance=5.0,
            emergency_stop_enabled=True
        )

        assert config.require_confirmation_for_movement is True
        assert config.max_movement_distance == 5.0
        assert config.emergency_stop_enabled is True


class TestCreateConfig:
    """Tests for create_config convenience function."""

    def test_create_dummy_config(self):
        """Test creating dummy config."""
        config = create_config(provider="dummy")

        assert config.ai_provider.provider_type == AIProviderType.DUMMY

    def test_create_gemini_config(self):
        """Test creating Gemini config."""
        config = create_config(provider="gemini", api_key="test-key")

        assert config.ai_provider.provider_type == AIProviderType.GEMINI
        assert config.ai_provider.api_key == "test-key"

    def test_create_config_with_kwargs(self):
        """Test creating config with additional kwargs."""
        config = create_config(
            provider="dummy",
            robot_name="CustomBot",
            debug_mode=True
        )

        assert config.robot_name == "CustomBot"
        assert config.debug_mode is True

    def test_create_config_with_model(self):
        """Test creating config with custom model."""
        config = create_config(
            provider="openai",
            model_name="gpt-4o"
        )

        assert config.ai_provider.model_name == "gpt-4o"
