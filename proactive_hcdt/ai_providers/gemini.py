"""
Google Gemini AI provider implementation.

This is the primary AI provider for the framework, supporting Google's
Gemini models for natural language understanding and generation.
"""

from typing import Any

from proactive_hcdt.ai_providers.base import (
    AIMessage,
    AIProvider,
    AIResponse,
    MessageRole,
    ToolCall,
)


class GeminiAIProvider(AIProvider):
    """
    Google Gemini AI provider.

    Provides integration with Google's Gemini models for
    multi-modal AI capabilities in robotic assistance.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: str | None = None,
    ):
        """
        Initialize the Gemini AI provider.

        Args:
            model_name: The Gemini model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash").
            api_key: Google AI API key. If not provided, will look for GOOGLE_API_KEY env var.
        """
        super().__init__(model_name, api_key)
        self._client = None
        self._model = None

    def _ensure_client(self) -> None:
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini provider. "
                    "Install it with: pip install google-generativeai"
                )

            if self.api_key:
                genai.configure(api_key=self.api_key)

            self._client = genai
            self._model = genai.GenerativeModel(self.model_name)

    async def generate(
        self,
        messages: list[AIMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """
        Generate a response using the Gemini model.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in the response.

        Returns:
            AIResponse containing the generated content and any tool calls.
        """
        self._ensure_client()

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)

        # Build generation config
        generation_config = {"temperature": temperature}
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Format tools if provided
        formatted_tools = None
        if tools:
            formatted_tools = self.format_tools(tools)

        try:
            # Create chat or direct generation based on message history
            if len(gemini_messages) > 1:
                chat = self._model.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"],
                    generation_config=generation_config,
                    tools=formatted_tools,
                )
            else:
                response = await self._model.generate_content_async(
                    gemini_messages[0]["parts"] if gemini_messages else "",
                    generation_config=generation_config,
                    tools=formatted_tools,
                )

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            # Return error response
            return AIResponse(
                content=f"Error generating response: {str(e)}",
                tool_calls=[],
                finish_reason="error",
            )

    def _convert_messages(self, messages: list[AIMessage]) -> list[dict[str, Any]]:
        """Convert AIMessage list to Gemini format."""
        gemini_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Gemini doesn't have a system role, prepend to first user message
                continue

            role = "user" if msg.role in [MessageRole.USER, MessageRole.TOOL] else "model"

            gemini_msg = {"role": role, "parts": [msg.content]}
            gemini_messages.append(gemini_msg)

        return gemini_messages

    def _parse_response(self, response: Any) -> AIResponse:
        """Parse Gemini response into AIResponse."""
        tool_calls = []
        content = ""

        try:
            candidate = response.candidates[0]
            content_parts = candidate.content.parts

            for part in content_parts:
                if hasattr(part, "text"):
                    content += part.text
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"gemini_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args),
                        )
                    )

            finish_reason = (
                str(candidate.finish_reason) if hasattr(candidate, "finish_reason") else None
            )

        except (IndexError, AttributeError) as e:
            content = f"Error parsing response: {str(e)}"
            finish_reason = "error"

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response,
            finish_reason=finish_reason,
        )

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format tools for Gemini's function calling format.

        Args:
            tools: List of tool definitions in standard format.

        Returns:
            List of tool definitions in Gemini format.
        """
        gemini_tools = []

        for tool in tools:
            gemini_tool = {
                "function_declarations": [
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                ]
            }
            gemini_tools.append(gemini_tool)

        return gemini_tools

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"
