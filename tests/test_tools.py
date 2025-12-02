"""Tests for the tool system."""

import pytest

from proactive_hcdt.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolResult
from proactive_hcdt.tools.examples import (
    CommunicationTool,
    ManipulationTool,
    MovementTool,
    PerceptionTool,
)


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_to_json_schema_basic(self):
        """Test converting basic parameter to JSON schema."""
        param = ToolParameter(
            name="test_param",
            type=ToolParameterType.STRING,
            description="A test parameter",
            required=True
        )

        schema = param.to_json_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "A test parameter"

    def test_to_json_schema_with_enum(self):
        """Test converting parameter with enum to JSON schema."""
        param = ToolParameter(
            name="direction",
            type=ToolParameterType.STRING,
            description="Direction",
            enum=["forward", "backward"]
        )

        schema = param.to_json_schema()

        assert schema["enum"] == ["forward", "backward"]

    def test_to_json_schema_with_default(self):
        """Test converting parameter with default to JSON schema."""
        param = ToolParameter(
            name="speed",
            type=ToolParameterType.NUMBER,
            description="Speed",
            required=False,
            default=0.5
        )

        schema = param.to_json_schema()

        assert schema["default"] == 0.5


class TestToolResult:
    """Tests for ToolResult."""

    def test_to_message_success(self):
        """Test converting successful result to message."""
        result = ToolResult(success=True, data={"status": "ok"})
        message = result.to_message()
        assert "status" in message
        assert "ok" in message

    def test_to_message_success_string(self):
        """Test converting successful string result to message."""
        result = ToolResult(success=True, data="Done")
        assert result.to_message() == "Done"

    def test_to_message_success_none(self):
        """Test converting successful result with no data."""
        result = ToolResult(success=True)
        assert result.to_message() == "Success"

    def test_to_message_error(self):
        """Test converting error result to message."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.to_message() == "Error: Something went wrong"


class TestBaseTool:
    """Tests for BaseTool base class."""

    def test_subclass_validation(self):
        """Test that subclasses must define name and description."""
        with pytest.raises(TypeError):
            class InvalidTool(BaseTool):
                pass

            InvalidTool()

    def test_valid_subclass(self):
        """Test that valid subclasses work correctly."""
        class ValidTool(BaseTool):
            name = "valid_tool"
            description = "A valid tool"
            parameters = []

            async def execute(self, **kwargs):
                return ToolResult(success=True, data="executed")

        tool = ValidTool()
        assert tool.name == "valid_tool"

    def test_to_schema(self):
        """Test converting tool to schema."""
        class SchemaTool(BaseTool):
            name = "schema_tool"
            description = "Tool for schema testing"
            parameters = [
                ToolParameter(
                    name="input",
                    type=ToolParameterType.STRING,
                    description="Input value",
                    required=True
                ),
                ToolParameter(
                    name="option",
                    type=ToolParameterType.BOOLEAN,
                    description="An option",
                    required=False
                )
            ]

            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = SchemaTool()
        schema = tool.to_schema()

        assert schema["name"] == "schema_tool"
        assert schema["description"] == "Tool for schema testing"
        assert "input" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["input"]


class TestMovementTool:
    """Tests for MovementTool."""

    @pytest.fixture
    def tool(self):
        return MovementTool()

    @pytest.mark.asyncio
    async def test_move_direction(self, tool):
        """Test moving in a direction."""
        result = await tool.execute(direction="forward", distance=2.0)

        assert result.success
        assert result.data["message"] == "Moved 2.0m forward"
        assert "new_position" in result.data

    @pytest.mark.asyncio
    async def test_move_to_position(self, tool):
        """Test moving to absolute position."""
        target = {"x": 1.0, "y": 2.0, "z": 0.0}
        result = await tool.execute(target_position=target)

        assert result.success
        assert result.data["new_position"] == target

    @pytest.mark.asyncio
    async def test_move_invalid_speed(self, tool):
        """Test invalid speed parameter."""
        result = await tool.execute(direction="forward", speed=5.0)

        assert not result.success
        assert "speed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_move_missing_params(self, tool):
        """Test missing required parameters."""
        result = await tool.execute()

        assert not result.success


class TestPerceptionTool:
    """Tests for PerceptionTool."""

    @pytest.fixture
    def tool(self):
        return PerceptionTool()

    @pytest.mark.asyncio
    async def test_object_detection(self, tool):
        """Test object detection scan."""
        result = await tool.execute(perception_type="object_detection")

        assert result.success
        assert "objects" in result.data

    @pytest.mark.asyncio
    async def test_person_detection(self, tool):
        """Test person detection scan."""
        result = await tool.execute(perception_type="person_detection")

        assert result.success
        assert "persons" in result.data

    @pytest.mark.asyncio
    async def test_full_scan(self, tool):
        """Test full environment scan."""
        result = await tool.execute(perception_type="full_scan")

        assert result.success
        assert "objects" in result.data
        assert "persons" in result.data
        assert "environment" in result.data


class TestCommunicationTool:
    """Tests for CommunicationTool."""

    @pytest.fixture
    def tool(self):
        return CommunicationTool()

    @pytest.mark.asyncio
    async def test_communicate(self, tool):
        """Test basic communication."""
        result = await tool.execute(message="Hello, human!")

        assert result.success
        assert result.data["communicated"]

    @pytest.mark.asyncio
    async def test_communicate_with_options(self, tool):
        """Test communication with options."""
        result = await tool.execute(
            message="Hello!",
            method="both",
            emotion="friendly"
        )

        assert result.success
        assert result.metadata["method"] == "both"
        assert result.metadata["emotion"] == "friendly"

    @pytest.mark.asyncio
    async def test_communicate_missing_message(self, tool):
        """Test communication without message."""
        result = await tool.execute()

        assert not result.success

    def test_message_history(self, tool):
        """Test message history tracking."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            tool.execute(message="First")
        )
        asyncio.get_event_loop().run_until_complete(
            tool.execute(message="Second")
        )

        history = tool.get_message_history()
        assert len(history) == 2


class TestManipulationTool:
    """Tests for ManipulationTool."""

    @pytest.fixture
    def tool(self):
        return ManipulationTool()

    @pytest.mark.asyncio
    async def test_grab(self, tool):
        """Test grabbing an object."""
        result = await tool.execute(action="grab", target_object="cup")

        assert result.success
        assert tool.is_holding
        assert tool.held_object == "cup"

    @pytest.mark.asyncio
    async def test_release(self, tool):
        """Test releasing an object."""
        await tool.execute(action="grab", target_object="cup")
        result = await tool.execute(action="release", target_object="cup")

        assert result.success
        assert not tool.is_holding

    @pytest.mark.asyncio
    async def test_grab_while_holding(self, tool):
        """Test grabbing while already holding."""
        await tool.execute(action="grab", target_object="cup")
        result = await tool.execute(action="grab", target_object="plate")

        assert not result.success
        assert "already holding" in result.error.lower()

    @pytest.mark.asyncio
    async def test_place_without_position(self, tool):
        """Test placing without position."""
        await tool.execute(action="grab", target_object="cup")
        result = await tool.execute(action="place", target_object="cup")

        assert not result.success
        assert "position" in result.error.lower()

    @pytest.mark.asyncio
    async def test_push(self, tool):
        """Test pushing an object."""
        result = await tool.execute(
            action="push",
            target_object="box",
            force=0.3
        )

        assert result.success
        assert result.data["action"] == "push"
