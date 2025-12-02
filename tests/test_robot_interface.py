"""Tests for robot interface."""

import pytest

from proactive_hcdt.robot_interface import DummyRobotInterface
from proactive_hcdt.robot_interface.base import RobotState


class TestRobotState:
    """Tests for RobotState."""

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = RobotState(
            position={"x": 1.0, "y": 2.0, "z": 0.0},
            orientation={"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            battery_level=0.8,
            is_moving=False,
            is_holding_object=True,
            held_object="cup"
        )

        state_dict = state.to_dict()

        assert state_dict["position"]["x"] == 1.0
        assert state_dict["battery_level"] == 0.8
        assert state_dict["held_object"] == "cup"


class TestDummyRobotInterface:
    """Tests for DummyRobotInterface."""

    @pytest.fixture
    def robot(self):
        return DummyRobotInterface(name="TestBot")

    @pytest.mark.asyncio
    async def test_initialize(self, robot):
        """Test robot initialization."""
        result = await robot.initialize()

        assert result is True

    @pytest.mark.asyncio
    async def test_shutdown(self, robot):
        """Test robot shutdown."""
        await robot.initialize()
        result = await robot.shutdown()

        assert result is True

    @pytest.mark.asyncio
    async def test_get_state(self, robot):
        """Test getting robot state."""
        await robot.initialize()
        state = await robot.get_state()

        assert isinstance(state, RobotState)
        assert state.battery_level > 0

    @pytest.mark.asyncio
    async def test_move_to(self, robot):
        """Test moving to position."""
        await robot.initialize()
        target = {"x": 1.0, "y": 2.0, "z": 0.0}
        result = await robot.move_to(target)

        assert result is True
        state = await robot.get_state()
        assert state.position == target

    @pytest.mark.asyncio
    async def test_move_without_init(self, robot):
        """Test moving without initialization."""
        result = await robot.move_to({"x": 1.0, "y": 1.0, "z": 0.0})

        assert result is False

    @pytest.mark.asyncio
    async def test_rotate(self, robot):
        """Test rotation."""
        await robot.initialize()
        result = await robot.rotate(90.0, axis="yaw")

        assert result is True
        state = await robot.get_state()
        assert state.orientation["yaw"] == 90.0

    @pytest.mark.asyncio
    async def test_rotate_invalid_axis(self, robot):
        """Test rotation with invalid axis."""
        await robot.initialize()
        result = await robot.rotate(90.0, axis="invalid")

        assert result is False

    @pytest.mark.asyncio
    async def test_stop(self, robot):
        """Test stopping."""
        result = await robot.stop()

        assert result is True

    @pytest.mark.asyncio
    async def test_speak(self, robot):
        """Test speech."""
        await robot.initialize()
        result = await robot.speak("Hello, human!")

        assert result is True

    @pytest.mark.asyncio
    async def test_display(self, robot):
        """Test display."""
        await robot.initialize()
        result = await robot.display("Status: OK")

        assert result is True

    @pytest.mark.asyncio
    async def test_grab_and_release(self, robot):
        """Test grabbing and releasing."""
        await robot.initialize()

        grab_result = await robot.grab("cup")
        assert grab_result is True

        state = await robot.get_state()
        assert state.is_holding_object
        assert state.held_object == "cup"

        release_result = await robot.release()
        assert release_result is True

        state = await robot.get_state()
        assert not state.is_holding_object

    @pytest.mark.asyncio
    async def test_grab_while_holding(self, robot):
        """Test grabbing while already holding."""
        await robot.initialize()
        await robot.grab("cup")

        result = await robot.grab("plate")

        assert result is False

    def test_robot_name(self, robot):
        """Test robot name property."""
        assert robot.robot_name == "TestBot"

    def test_capabilities(self, robot):
        """Test capabilities property."""
        caps = robot.capabilities
        assert "movement" in caps
        assert "manipulation" in caps

    @pytest.mark.asyncio
    async def test_set_position(self, robot):
        """Test directly setting position for testing."""
        robot.set_position({"x": 5.0, "y": 5.0, "z": 0.0})

        state = await robot.get_state()
        assert state.position["x"] == 5.0

    @pytest.mark.asyncio
    async def test_set_battery_level(self, robot):
        """Test setting battery level."""
        robot.set_battery_level(0.5)

        state = await robot.get_state()
        assert state.battery_level == 0.5

    @pytest.mark.asyncio
    async def test_trigger_and_clear_error(self, robot):
        """Test triggering and clearing error state."""
        robot.trigger_error("Test error")

        state = await robot.get_state()
        assert state.error_state == "Test error"

        robot.clear_error()
        state = await robot.get_state()
        assert state.error_state is None
        assert state.error_state is None
