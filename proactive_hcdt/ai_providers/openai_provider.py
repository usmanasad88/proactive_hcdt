"""
OpenAI AI provider implementation.

Supports OpenAI's GPT models for natural language understanding
and generation with function calling capabilities.
"""

import json
from typing import Any

from proactive_hcdt.ai_providers.base import (
    AIMessage,
    AIProvider,
    AIResponse,
    ToolCall,
)


class OpenAIProvider(AIProvider):
    """
    OpenAI GPT AI provider.

    Provides integration with OpenAI's GPT models for
    natural language understanding and tool use.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        api_key: str | None = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model_name: The OpenAI model to use (e.g., "gpt-4-turbo-preview", "gpt-4o").
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
        """
        super().__init__(model_name, api_key)
        self._client = None

    def _ensure_client(self) -> None:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI provider. "
                    "Install it with: pip install openai"
                )

            self._client = AsyncOpenAI(api_key=self.api_key) if self.api_key else AsyncOpenAI()

    async def generate(
        self,
        messages: list[AIMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Generate a response using an OpenAI model.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in the response.

        Returns:
            AIResponse containing the generated content and any tool calls.
        """
        self._ensure_client()

        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"

        try:
            response = await self._client.chat.completions.create(**params)
            return self._parse_response(response)

        except Exception as e:
            return AIResponse(
                content=f"Error generating response: {str(e)}",
                tool_calls=[],
                finish_reason="error",
            )

    def _convert_messages(self, messages: list[AIMessage]) -> list[dict[str, Any]]:
        """Convert AIMessage list to OpenAI format."""
        openai_messages = []

        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls

            openai_messages.append(openai_msg)

        return openai_messages

    def _parse_response(self, response: Any) -> AIResponse:
        """Parse OpenAI response into AIResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return AIResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            raw_response=response,
            finish_reason=choice.finish_reason,
        )

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for OpenAI's function calling format.

        Args:
            tools: List of tool definitions in standard format.

        Returns:
            List of tool definitions in OpenAI format.
        """
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"
