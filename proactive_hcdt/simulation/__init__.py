"""
Simulation environment for human-AI collaborative tasks.

A gamified 2D environment where a human-controlled agent and an AI-controlled 
agent work together to complete tasks. Features compliant push dynamics
similar to the push-T environment.

Modules:
- physics: PyMunk-based physics simulation (PD control)
- pymunk_renderer: Pygame renderer for PyMunk bodies
- ai_actions: High-level action primitives for AI/VLM control
"""

from .world import World, WorldConfig, GoalZone
from .agents import Agent, HumanAgent, AIAgent
from .shapes import (
    PushableObject, ShapeType,
    create_box, create_l_shape, create_t_shape
)
from .renderer import Renderer
from .physics import PhysicsWorld, PhysicsConfig
from .pymunk_renderer import PymunkRenderer
from .ai_actions import (
    AIActionPrimitives,
    WorldObservation,
    ObjectInfo,
    GoalInfo,
    ActionResult,
    ActionStatus,
)

__all__ = [
    # Legacy world (pygame-only)
    "World",
    "WorldConfig",
    "GoalZone",
    "Agent",
    "HumanAgent",
    "AIAgent",
    "PushableObject",
    "ShapeType",
    "create_box",
    "create_l_shape",
    "create_t_shape",
    "Renderer",
    # PyMunk physics
    "PhysicsWorld",
    "PhysicsConfig",
    "PymunkRenderer",
    # AI primitives
    "AIActionPrimitives",
    "WorldObservation",
    "ObjectInfo",
    "GoalInfo",
    "ActionResult",
    "ActionStatus",
]
