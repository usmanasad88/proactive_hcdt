"""
Abstract base class for robot hardware interfaces.

This module defines the interface that all robot platforms must implement
to integrate with the proactive assistance framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RobotState:
    """Current state of the robot."""

    position: dict[str, float]  # x, y, z coordinates
    orientation: dict[str, float]  # roll, pitch, yaw
    battery_level: float  # 0.0 to 1.0
    is_moving: bool
    is_holding_object: bool
    held_object: str | None = None
    error_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "position": self.position,
            "orientation": self.orientation,
            "battery_level": self.battery_level,
            "is_moving": self.is_moving,
            "is_holding_object": self.is_holding_object,
            "held_object": self.held_object,
            "error_state": self.error_state,
        }


class RobotInterface(ABC):
    """
    Abstract base class for robot hardware interfaces.

    All robot platforms should implement this interface to work
    with the proactive assistance framework.
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the robot hardware.

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Safely shutdown the robot hardware.

        Returns:
            True if shutdown successful, False otherwise.
        """
        pass

    @abstractmethod
    async def get_state(self) -> RobotState:
        """
        Get the current robot state.

        Returns:
            RobotState object with current status.
        """
        pass

    @abstractmethod
    async def move_to(
        self,
        position: dict[str, float],
        speed: float = 0.5,
    ) -> bool:
        """
        Move the robot to a target position.

        Args:
            position: Target position as {x, y, z} dict.
            speed: Movement speed (0.0 to 1.0).

        Returns:
            True if movement successful, False otherwise.
        """
        pass

    @abstractmethod
    async def rotate(
        self,
        angle: float,
        axis: str = "yaw",
    ) -> bool:
        """
        Rotate the robot by a specified angle.

        Args:
            angle: Rotation angle in degrees.
            axis: Rotation axis (roll, pitch, yaw).

        Returns:
            True if rotation successful, False otherwise.
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop all robot movement immediately.

        Returns:
            True if stop successful, False otherwise.
        """
        pass

    @abstractmethod
    async def speak(self, text: str, language: str = "en-US") -> bool:
        """
        Speak text using text-to-speech.

        Args:
            text: Text to speak.
            language: Language code.

        Returns:
            True if speech successful, False otherwise.
        """
        pass

    @abstractmethod
    async def display(self, content: str | dict[str, Any]) -> bool:
        """
        Display content on the robot's screen.

        Args:
            content: Text or structured content to display.

        Returns:
            True if display successful, False otherwise.
        """
        pass

    @abstractmethod
    async def grab(self, target: str) -> bool:
        """
        Grab an object with the gripper.

        Args:
            target: Object identifier or description.

        Returns:
            True if grab successful, False otherwise.
        """
        pass

    @abstractmethod
    async def release(self) -> bool:
        """
        Release the currently held object.

        Returns:
            True if release successful, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def robot_name(self) -> str:
        """Return the robot's name/identifier."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """Return list of robot capabilities."""
        pass
