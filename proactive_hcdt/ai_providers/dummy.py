"""
Dummy AI provider for testing and development.

This provider simulates AI responses without requiring actual API calls,
making it useful for testing the framework and developing new features.
"""

import uuid
from typing import Any

from proactive_hcdt.ai_providers.base import AIMessage, AIProvider, AIResponse, ToolCall


class DummyAIProvider(AIProvider):
    """
    Dummy AI provider for testing purposes.

    This provider returns predefined responses based on patterns in the input,
    and can simulate tool calls for testing the tool execution pipeline.
    """

    def __init__(
        self,
        model_name: str = "dummy-model-v1",
        api_key: str | None = None,
        responses: dict[str, str] | None = None,
    ):
        """
        Initialize the dummy AI provider.

        Args:
            model_name: Name of the dummy model.
            api_key: Ignored, included for interface compatibility.
            responses: Optional dictionary mapping input patterns to responses.
        """
        super().__init__(model_name, api_key)
        self.responses = responses or {}
        self._call_count = 0

    async def generate(
        self,
        messages: list[AIMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Generate a dummy response based on the input messages.

        The dummy provider can:
        - Return predefined responses based on patterns
        - Simulate tool calls if tools are provided and input suggests a command
        - Return a generic acknowledgment otherwise
        """
        self._call_count += 1

        # Get the last user message
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role.value == "user":
                last_user_message = msg.content.lower()
                break

        # Check for predefined responses
        for pattern, response in self.responses.items():
            if pattern.lower() in last_user_message:
                return AIResponse(
                    content=response,
                    tool_calls=[],
                    finish_reason="stop",
                )

        # Simulate tool calls if tools are provided and input suggests a command
        if tools and self._should_use_tool(last_user_message, tools):
            tool_call = self._generate_tool_call(last_user_message, tools)
            if tool_call:
                return AIResponse(
                    content="",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )

        # Default response
        return AIResponse(
            content=f"Acknowledged: {last_user_message[:50]}... (dummy response #{self._call_count})",
            tool_calls=[],
            finish_reason="stop",
        )

    def _should_use_tool(self, message: str, tools: list[dict[str, Any]]) -> bool:
        """Check if the message suggests using a tool."""
        action_words = [
            "move",
            "go",
            "turn",
            "grab",
            "pick",
            "say",
            "speak",
            "look",
            "scan",
            "detect",
            "find",
            "help",
            "assist",
            "fetch",
            "bring",
        ]
        return any(word in message for word in action_words)

    def _generate_tool_call(
        self, message: str, tools: list[dict[str, Any]]
    ) -> ToolCall | None:
        """Generate a simulated tool call based on the message."""
        # Try to match the message to an available tool
        for tool in tools:
            tool_name = tool.get("name", "")

            # Simple matching logic based on tool name
            if any(keyword in message for keyword in tool_name.lower().split("_")):
                # Generate dummy arguments
                arguments = self._generate_dummy_arguments(tool)
                return ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=tool_name,
                    arguments=arguments,
                )

        # If no specific match, use the first available tool
        if tools:
            tool = tools[0]
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=tool.get("name", "unknown"),
                arguments=self._generate_dummy_arguments(tool),
            )

        return None

    def _generate_dummy_arguments(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Generate dummy arguments for a tool based on its parameters."""
        arguments = {}
        parameters = tool.get("parameters", {}).get("properties", {})

        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "string")

            if param_type == "string":
                arguments[param_name] = f"dummy_{param_name}"
            elif param_type == "number":
                arguments[param_name] = 1.0
            elif param_type == "integer":
                arguments[param_name] = 1
            elif param_type == "boolean":
                arguments[param_name] = True
            elif param_type == "array":
                arguments[param_name] = []
            elif param_type == "object":
                arguments[param_name] = {}

        return arguments

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for the dummy provider (pass-through).

        The dummy provider accepts any tool format, so this is a simple pass-through.
        """
        return tools

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "dummy"

    def set_response(self, pattern: str, response: str) -> None:
        """
        Set a predefined response for a specific input pattern.

        Args:
            pattern: The input pattern to match.
            response: The response to return when the pattern is matched.
        """
        self.responses[pattern] = response

    def reset(self) -> None:
        """Reset the provider state."""
        self._call_count = 0
        self.responses.clear()
