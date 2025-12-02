"""
Anthropic Claude AI provider implementation.

Supports Anthropic's Claude models for natural language understanding
and generation with tool use capabilities.
"""

from typing import Any

from proactive_hcdt.ai_providers.base import (
    AIMessage,
    AIProvider,
    AIResponse,
    MessageRole,
    ToolCall,
)


class AnthropicProvider(AIProvider):
    """
    Anthropic Claude AI provider.

    Provides integration with Anthropic's Claude models for
    natural language understanding and tool use.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model_name: The Claude model to use (e.g., "claude-3-5-sonnet-20241022").
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
        """
        super().__init__(model_name, api_key)
        self._client = None

    def _ensure_client(self) -> None:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic provider. "
                    "Install it with: pip install anthropic"
                )

            self._client = (
                AsyncAnthropic(api_key=self.api_key) if self.api_key else AsyncAnthropic()
            )

    async def generate(
        self,
        messages: list[AIMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Generate a response using a Claude model.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in the response.

        Returns:
            AIResponse containing the generated content and any tool calls.
        """
        self._ensure_client()

        # Extract system message and convert others to Claude format
        system_message, claude_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self.format_tools(tools)

        try:
            response = await self._client.messages.create(**params)
            return self._parse_response(response)

        except Exception as e:
            return AIResponse(
                content=f"Error generating response: {str(e)}",
                tool_calls=[],
                finish_reason="error",
            )

    def _convert_messages(
        self, messages: list[AIMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert AIMessage list to Claude format, extracting system message."""
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
                continue

            role = msg.role.value
            if role == "tool":
                # Claude uses tool_result for tool responses
                claude_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            else:
                claude_messages.append({"role": role, "content": msg.content})

        return system_message, claude_messages

    def _parse_response(self, response: Any) -> AIResponse:
        """Parse Claude response into AIResponse."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response,
            finish_reason=response.stop_reason,
        )

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for Claude's tool use format.

        Args:
            tools: List of tool definitions in standard format.

        Returns:
            List of tool definitions in Claude format.
        """
        claude_tools = []

        for tool in tools:
            claude_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
            claude_tools.append(claude_tool)

        return claude_tools

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"
