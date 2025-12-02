"""Tests for AI providers."""

import pytest

from proactive_hcdt.ai_providers import AIMessage, AIResponse, DummyAIProvider
from proactive_hcdt.ai_providers.base import MessageRole


class TestDummyAIProvider:
    """Tests for the DummyAIProvider."""

    @pytest.fixture
    def provider(self):
        """Create a DummyAIProvider instance."""
        return DummyAIProvider()

    @pytest.mark.asyncio
    async def test_generate_basic_response(self, provider):
        """Test generating a basic response."""
        messages = [
            AIMessage(role=MessageRole.USER, content="Hello, robot!")
        ]

        response = await provider.generate(messages)

        assert isinstance(response, AIResponse)
        assert response.content is not None
        assert "Hello" in response.content or "dummy response" in response.content

    @pytest.mark.asyncio
    async def test_generate_with_predefined_response(self, provider):
        """Test generating with predefined response patterns."""
        provider.set_response("greet", "Hello, human!")

        messages = [
            AIMessage(role=MessageRole.USER, content="Please greet me")
        ]

        response = await provider.generate(messages)

        assert response.content == "Hello, human!"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, provider):
        """Test generating with tools triggers tool calls for action words."""
        tools = [
            {
                "name": "move_robot",
                "description": "Move the robot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string"}
                    }
                }
            }
        ]

        messages = [
            AIMessage(role=MessageRole.USER, content="Move forward")
        ]

        response = await provider.generate(messages, tools=tools)

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "move_robot"

    def test_format_tools(self, provider):
        """Test tool formatting (pass-through for dummy)."""
        tools = [{"name": "test_tool", "description": "A test tool"}]
        formatted = provider.format_tools(tools)
        assert formatted == tools

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.provider_name == "dummy"

    def test_reset(self, provider):
        """Test resetting the provider."""
        provider.set_response("test", "response")
        provider._call_count = 5

        provider.reset()

        assert provider._call_count == 0
        assert len(provider.responses) == 0


class TestAIMessage:
    """Tests for AIMessage."""

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = AIMessage(
            role=MessageRole.USER,
            content="Hello",
            tool_call_id="call_123"
        )

        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert result["tool_call_id"] == "call_123"

    def test_to_dict_minimal(self):
        """Test converting minimal message to dictionary."""
        msg = AIMessage(role=MessageRole.ASSISTANT, content="Hi")

        result = msg.to_dict()

        assert result["role"] == "assistant"
        assert result["content"] == "Hi"
        assert "tool_call_id" not in result


class TestAIResponse:
    """Tests for AIResponse."""

    def test_has_tool_calls_true(self):
        """Test has_tool_calls when there are tool calls."""
        from proactive_hcdt.ai_providers.base import ToolCall

        response = AIResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="test", arguments={})]
        )

        assert response.has_tool_calls is True

    def test_has_tool_calls_false(self):
        """Test has_tool_calls when there are no tool calls."""
        response = AIResponse(content="Hello")

        assert response.has_tool_calls is False
