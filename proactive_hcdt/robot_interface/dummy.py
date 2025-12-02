"""
Dummy robot interface for testing and development.

This implementation simulates robot hardware without requiring
actual robot connections.
"""

import asyncio
from typing import Any

from proactive_hcdt.robot_interface.base import RobotInterface, RobotState


class DummyRobotInterface(RobotInterface):
    """
    Dummy robot interface for testing.

    Simulates all robot operations with configurable delays
    and success rates for testing different scenarios.
    """

    def __init__(
        self,
        name: str = "DummyBot",
        simulation_delay: float = 0.1,
        failure_rate: float = 0.0,
    ):
        """
        Initialize the dummy robot interface.

        Args:
            name: Name/identifier for the robot.
            simulation_delay: Delay in seconds for simulated operations.
            failure_rate: Probability of operation failure (0.0 to 1.0).
        """
        self._name = name
        self._simulation_delay = simulation_delay
        self._failure_rate = failure_rate

        self._initialized = False
        self._position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._battery_level = 1.0
        self._is_moving = False
        self._held_object: str | None = None
        self._error_state: str | None = None

        self._capabilities = [
            "movement",
            "rotation",
            "speech",
            "display",
            "manipulation",
        ]

    async def _simulate_operation(self) -> bool:
        """Simulate an operation with delay and potential failure."""
        await asyncio.sleep(self._simulation_delay)

        import random

        if random.random() < self._failure_rate:
            return False
        return True

    async def initialize(self) -> bool:
        """Initialize the dummy robot."""
        await asyncio.sleep(self._simulation_delay * 2)
        self._initialized = True
        self._error_state = None
        return True

    async def shutdown(self) -> bool:
        """Shutdown the dummy robot."""
        await asyncio.sleep(self._simulation_delay)
        self._initialized = False
        self._is_moving = False
        return True

    async def get_state(self) -> RobotState:
        """Get the current robot state."""
        return RobotState(
            position=self._position.copy(),
            orientation=self._orientation.copy(),
            battery_level=self._battery_level,
            is_moving=self._is_moving,
            is_holding_object=self._held_object is not None,
            held_object=self._held_object,
            error_state=self._error_state,
        )

    async def move_to(
        self,
        position: dict[str, float],
        speed: float = 0.5,
    ) -> bool:
        """Simulate movement to a target position."""
        if not self._initialized:
            self._error_state = "Robot not initialized"
            return False

        self._is_moving = True

        if await self._simulate_operation():
            self._position = position.copy()
            self._is_moving = False
            self._battery_level = max(0.0, self._battery_level - 0.01)
            return True
        else:
            self._is_moving = False
            self._error_state = "Movement failed"
            return False

    async def rotate(
        self,
        angle: float,
        axis: str = "yaw",
    ) -> bool:
        """Simulate rotation."""
        if not self._initialized:
            self._error_state = "Robot not initialized"
            return False

        if axis not in ["roll", "pitch", "yaw"]:
            self._error_state = f"Invalid rotation axis: {axis}"
            return False

        if await self._simulate_operation():
            self._orientation[axis] = (self._orientation[axis] + angle) % 360
            return True
        else:
            self._error_state = "Rotation failed"
            return False

    async def stop(self) -> bool:
        """Stop all movement."""
        self._is_moving = False
        return True

    async def speak(self, text: str, language: str = "en-US") -> bool:
        """Simulate speech output."""
        if not self._initialized:
            return False

        if await self._simulate_operation():
            # In real implementation, this would use TTS
            return True
        return False

    async def display(self, content: str | dict[str, Any]) -> bool:
        """Simulate display output."""
        if not self._initialized:
            return False

        if await self._simulate_operation():
            # In real implementation, this would update a display
            return True
        return False

    async def grab(self, target: str) -> bool:
        """Simulate grabbing an object."""
        if not self._initialized:
            self._error_state = "Robot not initialized"
            return False

        if self._held_object is not None:
            self._error_state = f"Already holding {self._held_object}"
            return False

        if await self._simulate_operation():
            self._held_object = target
            return True
        else:
            self._error_state = f"Failed to grab {target}"
            return False

    async def release(self) -> bool:
        """Simulate releasing an object."""
        if not self._initialized:
            self._error_state = "Robot not initialized"
            return False

        if self._held_object is None:
            self._error_state = "Not holding any object"
            return False

        if await self._simulate_operation():
            self._held_object = None
            return True
        else:
            self._error_state = "Failed to release object"
            return False

    @property
    def robot_name(self) -> str:
        """Return the robot name."""
        return self._name

    @property
    def capabilities(self) -> list[str]:
        """Return robot capabilities."""
        return self._capabilities.copy()

    def set_position(self, position: dict[str, float]) -> None:
        """Directly set position for testing."""
        self._position = position.copy()

    def set_battery_level(self, level: float) -> None:
        """Directly set battery level for testing."""
        self._battery_level = max(0.0, min(1.0, level))

    def trigger_error(self, error: str) -> None:
        """Trigger an error state for testing."""
        self._error_state = error

    def clear_error(self) -> None:
        """Clear error state."""
        self._error_state = None
