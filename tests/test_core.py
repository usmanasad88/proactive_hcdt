"""Tests for the core components (controller and registry)."""

import pytest

from proactive_hcdt.ai_providers import DummyAIProvider
from proactive_hcdt.core.controller import AIController
from proactive_hcdt.core.tool_registry import ToolRegistry
from proactive_hcdt.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolResult
from proactive_hcdt.tools.examples import CommunicationTool, MovementTool


class SimpleTool(BaseTool):
    """Simple tool for testing."""

    name = "simple_tool"
    description = "A simple test tool"
    parameters = [
        ToolParameter(
            name="value",
            type=ToolParameterType.STRING,
            description="A test value",
            required=True
        )
    ]

    async def execute(self, **kwargs):
        value = kwargs.get("value", "")
        return ToolResult(success=True, data={"received": value})


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def simple_tool(self):
        return SimpleTool()

    def test_register(self, registry, simple_tool):
        """Test registering a tool."""
        registry.register(simple_tool)

        assert simple_tool.name in registry
        assert len(registry) == 1

    def test_register_duplicate(self, registry, simple_tool):
        """Test registering duplicate tool raises error."""
        registry.register(simple_tool)

        with pytest.raises(ValueError):
            registry.register(simple_tool)

    def test_register_many(self, registry):
        """Test registering multiple tools."""
        tools = [MovementTool(), CommunicationTool()]
        registry.register_many(tools)

        assert len(registry) == 2
        assert "move_robot" in registry
        assert "communicate" in registry

    def test_unregister(self, registry, simple_tool):
        """Test unregistering a tool."""
        registry.register(simple_tool)
        result = registry.unregister(simple_tool.name)

        assert result is True
        assert simple_tool.name not in registry

    def test_unregister_nonexistent(self, registry):
        """Test unregistering nonexistent tool."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get(self, registry, simple_tool):
        """Test getting a tool."""
        registry.register(simple_tool)
        retrieved = registry.get(simple_tool.name)

        assert retrieved is simple_tool

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent tool."""
        result = registry.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_execute(self, registry, simple_tool):
        """Test executing a tool."""
        registry.register(simple_tool)
        result = await registry.execute("simple_tool", value="test")

        assert result.success
        assert result.data["received"] == "test"

    @pytest.mark.asyncio
    async def test_execute_nonexistent(self, registry):
        """Test executing nonexistent tool."""
        result = await registry.execute("nonexistent")

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_args(self, registry, simple_tool):
        """Test executing with missing required argument."""
        registry.register(simple_tool)
        result = await registry.execute("simple_tool")

        assert not result.success
        assert "missing" in result.error.lower()

    def test_list_tools(self, registry):
        """Test listing tool names."""
        registry.register_many([MovementTool(), CommunicationTool()])
        names = registry.list_tools()

        assert "move_robot" in names
        assert "communicate" in names

    def test_get_schemas(self, registry, simple_tool):
        """Test getting tool schemas."""
        registry.register(simple_tool)
        schemas = registry.get_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "simple_tool"

    def test_clear(self, registry, simple_tool):
        """Test clearing registry."""
        registry.register(simple_tool)
        registry.clear()

        assert len(registry) == 0

    def test_iteration(self, registry):
        """Test iterating over registry."""
        tools = [MovementTool(), CommunicationTool()]
        registry.register_many(tools)

        tool_list = list(registry)
        assert len(tool_list) == 2


class TestAIController:
    """Tests for AIController."""

    @pytest.fixture
    def provider(self):
        return DummyAIProvider()

    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register_many([MovementTool(), CommunicationTool()])
        return reg

    @pytest.fixture
    def controller(self, provider, registry):
        return AIController(
            ai_provider=provider,
            tool_registry=registry
        )

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.ai_provider is not None
        assert controller.tool_registry is not None
        assert len(controller.available_tools) == 2

    def test_default_system_prompt(self, controller):
        """Test default system prompt is set."""
        assert controller.system_prompt is not None
        assert "robot" in controller.system_prompt.lower()

    def test_custom_system_prompt(self, provider):
        """Test custom system prompt."""
        custom_prompt = "You are a test robot."
        controller = AIController(
            ai_provider=provider,
            system_prompt=custom_prompt
        )

        assert controller.system_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_process_simple(self, controller, provider):
        """Test processing simple input."""
        provider.set_response("hello", "Hi there!")

        response = await controller.process("hello")

        assert response == "Hi there!"

    @pytest.mark.asyncio
    async def test_process_with_tool_call(self, controller):
        """Test processing input that triggers tool call."""
        response = await controller.process("Move forward")

        assert response is not None
        # The dummy provider should have triggered a tool call for "move"

    @pytest.mark.asyncio
    async def test_conversation_history(self, controller):
        """Test conversation history is maintained."""
        await controller.process("First message")

        history = controller.get_conversation_history()

        # Should have system + user + assistant messages
        assert len(history) >= 3

    @pytest.mark.asyncio
    async def test_clear_history(self, controller):
        """Test clearing conversation history."""
        await controller.process("Test message")

        controller.clear_history()
        history = controller.get_conversation_history()

        # Should only have system message
        assert len(history) == 1
        assert history[0]["role"] == "system"

    def test_add_context(self, controller):
        """Test adding context."""
        controller.add_context("User is standing nearby")

        history = controller.get_conversation_history()
        assert any("context" in msg["content"].lower() for msg in history)

    def test_register_tool(self, controller):
        """Test registering tool via controller."""
        initial_count = len(controller.available_tools)

        class NewTool(BaseTool):
            name = "new_tool"
            description = "A new tool"

            async def execute(self, **kwargs):
                return ToolResult(success=True)

        controller.register_tool(NewTool())

        assert len(controller.available_tools) == initial_count + 1

    def test_repr(self, controller):
        """Test string representation."""
        repr_str = repr(controller)
        assert "AIController" in repr_str
        assert "dummy" in repr_str


class TestAIControllerProactiveBehavior:
    """Tests for proactive behavior in AIController."""

    @pytest.fixture
    def controller_with_perception(self):
        """Create controller with perception tool."""
        from proactive_hcdt.tools.examples import PerceptionTool

        provider = DummyAIProvider()
        registry = ToolRegistry()
        registry.register(PerceptionTool())

        return AIController(ai_provider=provider, tool_registry=registry)

    @pytest.mark.asyncio
    async def test_proactive_scan(self, controller_with_perception):
        """Test proactive environment scanning."""
        result = await controller_with_perception.proactive_scan()

        assert result is not None

    @pytest.mark.asyncio
    async def test_proactive_scan_without_perception(self):
        """Test proactive scan without perception tool."""
        provider = DummyAIProvider()
        controller = AIController(ai_provider=provider)

        result = await controller.proactive_scan()

        assert result is None
