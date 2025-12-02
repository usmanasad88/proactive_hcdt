"""
Abstract base class for AI providers.

This module defines the interface that all AI providers must implement,
enabling seamless switching between different frontier AI models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class AIMessage:
    """Represents a message in the AI conversation."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        result = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class ToolCall:
    """Represents a tool call request from the AI."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AIResponse:
    """Response from an AI provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0


class AIProvider(ABC):
    """
    Abstract base class for AI providers.

    All AI providers (Gemini, Claude, OpenAI, etc.) must implement this interface
    to ensure consistent behavior across different models.
    """

    def __init__(self, model_name: str, api_key: str | None = None):
        """
        Initialize the AI provider.

        Args:
            model_name: The name/identifier of the model to use.
            api_key: Optional API key for authentication.
        """
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        messages: list[AIMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Generate a response from the AI model.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions in the provider's format.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in the response.

        Returns:
            AIResponse containing the generated content and any tool calls.
        """
        pass

    @abstractmethod
    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tool definitions for this specific provider.

        Args:
            tools: List of tool definitions in the standard format.

        Returns:
            List of tool definitions in the provider-specific format.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this AI provider."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
