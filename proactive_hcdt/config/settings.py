"""
Configuration settings for the proactive robotic assistance framework.

Provides dataclass-based configuration for easy setup and validation.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from proactive_hcdt.ai_providers.base import AIProvider
from proactive_hcdt.ai_providers.dummy import DummyAIProvider


class AIProviderType(str, Enum):
    """Supported AI provider types."""

    DUMMY = "dummy"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AIProviderConfig:
    """Configuration for an AI provider."""

    provider_type: AIProviderType = AIProviderType.DUMMY
    model_name: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

    def __post_init__(self):
        """Set default model names based on provider type."""
        if self.model_name is None:
            default_models = {
                AIProviderType.DUMMY: "dummy-model-v1",
                AIProviderType.GEMINI: "gemini-1.5-pro",
                AIProviderType.OPENAI: "gpt-4-turbo-preview",
                AIProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
            }
            self.model_name = default_models.get(self.provider_type, "default-model")

    def create_provider(self) -> AIProvider:
        """
        Create an AI provider instance based on this configuration.

        Returns:
            Configured AIProvider instance.

        Raises:
            ImportError: If the required package for the provider is not installed.
            ValueError: If the provider type is not supported.
        """
        if self.provider_type == AIProviderType.DUMMY:
            return DummyAIProvider(
                model_name=self.model_name or "dummy-model-v1",
                api_key=self.api_key,
            )

        elif self.provider_type == AIProviderType.GEMINI:
            from proactive_hcdt.ai_providers.gemini import GeminiAIProvider

            return GeminiAIProvider(
                model_name=self.model_name or "gemini-1.5-pro",
                api_key=self.api_key or os.getenv("GOOGLE_API_KEY"),
            )

        elif self.provider_type == AIProviderType.OPENAI:
            from proactive_hcdt.ai_providers.openai_provider import OpenAIProvider

            return OpenAIProvider(
                model_name=self.model_name or "gpt-4-turbo-preview",
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            )

        elif self.provider_type == AIProviderType.ANTHROPIC:
            from proactive_hcdt.ai_providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                model_name=self.model_name or "claude-3-5-sonnet-20241022",
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )

        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")


@dataclass
class FrameworkConfig:
    """
    Main configuration for the proactive robotic assistance framework.

    This configuration class holds all settings needed to initialize
    and run the framework.
    """

    # AI Provider settings
    ai_provider: AIProviderConfig = field(default_factory=AIProviderConfig)

    # Controller settings
    system_prompt: str | None = None
    max_tool_iterations: int = 10

    # Robot settings
    robot_name: str = "ProactiveBot"
    enable_proactive_scanning: bool = True
    scan_interval_seconds: float = 30.0

    # Safety settings
    require_confirmation_for_movement: bool = False
    max_movement_distance: float = 10.0  # meters
    emergency_stop_enabled: bool = True

    # Logging and debugging
    debug_mode: bool = False
    log_conversations: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FrameworkConfig":
        """
        Create a FrameworkConfig from a dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Configured FrameworkConfig instance.
        """
        ai_config = config_dict.get("ai_provider", {})
        if isinstance(ai_config, dict):
            if "provider_type" in ai_config:
                ai_config["provider_type"] = AIProviderType(ai_config["provider_type"])
            ai_provider = AIProviderConfig(**ai_config)
        else:
            ai_provider = ai_config

        return cls(
            ai_provider=ai_provider,
            system_prompt=config_dict.get("system_prompt"),
            max_tool_iterations=config_dict.get("max_tool_iterations", 10),
            robot_name=config_dict.get("robot_name", "ProactiveBot"),
            enable_proactive_scanning=config_dict.get("enable_proactive_scanning", True),
            scan_interval_seconds=config_dict.get("scan_interval_seconds", 30.0),
            require_confirmation_for_movement=config_dict.get(
                "require_confirmation_for_movement", False
            ),
            max_movement_distance=config_dict.get("max_movement_distance", 10.0),
            emergency_stop_enabled=config_dict.get("emergency_stop_enabled", True),
            debug_mode=config_dict.get("debug_mode", False),
            log_conversations=config_dict.get("log_conversations", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "ai_provider": {
                "provider_type": self.ai_provider.provider_type.value,
                "model_name": self.ai_provider.model_name,
                "temperature": self.ai_provider.temperature,
                "max_tokens": self.ai_provider.max_tokens,
            },
            "system_prompt": self.system_prompt,
            "max_tool_iterations": self.max_tool_iterations,
            "robot_name": self.robot_name,
            "enable_proactive_scanning": self.enable_proactive_scanning,
            "scan_interval_seconds": self.scan_interval_seconds,
            "require_confirmation_for_movement": self.require_confirmation_for_movement,
            "max_movement_distance": self.max_movement_distance,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "debug_mode": self.debug_mode,
            "log_conversations": self.log_conversations,
        }


def create_config(
    provider: Literal["dummy", "gemini", "openai", "anthropic"] = "dummy",
    model_name: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> FrameworkConfig:
    """
    Convenience function to create a framework configuration.

    Args:
        provider: AI provider type ("dummy", "gemini", "openai", "anthropic").
        model_name: Optional model name (uses default for provider if not specified).
        api_key: Optional API key (uses environment variable if not specified).
        **kwargs: Additional FrameworkConfig parameters.

    Returns:
        Configured FrameworkConfig instance.

    Example:
        ```python
        # Create a config with Gemini
        config = create_config(provider="gemini", api_key="your-key")

        # Create a config with custom settings
        config = create_config(
            provider="openai",
            model_name="gpt-4o",
            debug_mode=True,
            robot_name="MyBot"
        )
        ```
    """
    ai_config = AIProviderConfig(
        provider_type=AIProviderType(provider),
        model_name=model_name,
        api_key=api_key,
    )

    return FrameworkConfig(ai_provider=ai_config, **kwargs)
