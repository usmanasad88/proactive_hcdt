"""
Robot interface abstraction layer.

This module provides abstract interfaces for robot hardware,
allowing the framework to work with different robot platforms.
"""

from proactive_hcdt.robot_interface.base import RobotInterface
from proactive_hcdt.robot_interface.dummy import DummyRobotInterface

__all__ = ["RobotInterface", "DummyRobotInterface"]
