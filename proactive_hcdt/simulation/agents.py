"""
Agent definitions for the simulation environment.

Agents are entities that can move and interact within the world.
- HumanAgent: Controlled via keyboard input
- AIAgent: Controlled via API calls (for VLM integration)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum
import math


class AgentType(Enum):
    HUMAN = "human"
    AI = "ai"


@dataclass
class AgentState:
    """Immutable snapshot of agent state for observation."""
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    radius: float
    agent_type: AgentType
    name: str


@dataclass
class Agent:
    """Base agent class with position and movement."""
    
    name: str
    x: float = 0.0
    y: float = 0.0
    radius: float = 20.0
    color: Tuple[int, int, int] = (100, 100, 100)
    max_speed: float = 5.0
    agent_type: AgentType = AgentType.HUMAN
    
    # Movement state
    velocity_x: float = field(default=0.0, repr=False)
    velocity_y: float = field(default=0.0, repr=False)
    
    # Movement input (normalized direction)
    _move_x: float = field(default=0.0, repr=False)
    _move_y: float = field(default=0.0, repr=False)
    
    def set_movement(self, dx: float, dy: float) -> None:
        """Set movement direction (normalized)."""
        # Normalize if magnitude > 1
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude > 1.0:
            dx /= magnitude
            dy /= magnitude
        self._move_x = dx
        self._move_y = dy
    
    def stop(self) -> None:
        """Stop all movement."""
        self._move_x = 0.0
        self._move_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
    
    def update(self, dt: float, world_width: float, world_height: float) -> None:
        """Update agent position based on movement input."""
        # Apply movement
        self.velocity_x = self._move_x * self.max_speed
        self.velocity_y = self._move_y * self.max_speed
        
        # Update position
        self.x += self.velocity_x * dt * 60  # Scale by 60 for frame-rate independence
        self.y += self.velocity_y * dt * 60
        
        # Clamp to world bounds
        self.x = max(self.radius, min(world_width - self.radius, self.x))
        self.y = max(self.radius, min(world_height - self.radius, self.y))
    
    def get_state(self) -> AgentState:
        """Get immutable state snapshot."""
        return AgentState(
            x=self.x,
            y=self.y,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            radius=self.radius,
            agent_type=self.agent_type,
            name=self.name
        )
    
    def distance_to(self, other: "Agent") -> float:
        """Calculate distance to another agent."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


class HumanAgent(Agent):
    """Agent controlled by human keyboard input."""
    
    def __init__(
        self,
        name: str = "Human",
        x: float = 100.0,
        y: float = 300.0,
        color: Tuple[int, int, int] = (65, 105, 225),  # Royal blue
        **kwargs
    ):
        super().__init__(
            name=name,
            x=x,
            y=y,
            color=color,
            agent_type=AgentType.HUMAN,
            **kwargs
        )
    
    def handle_keyboard(self, keys_pressed: dict) -> None:
        """
        Process keyboard input for movement.
        
        Args:
            keys_pressed: Dict with keys 'w', 'a', 's', 'd' as bools
        """
        dx = 0.0
        dy = 0.0
        
        if keys_pressed.get('w') or keys_pressed.get('up'):
            dy -= 1.0
        if keys_pressed.get('s') or keys_pressed.get('down'):
            dy += 1.0
        if keys_pressed.get('a') or keys_pressed.get('left'):
            dx -= 1.0
        if keys_pressed.get('d') or keys_pressed.get('right'):
            dx += 1.0
        
        self.set_movement(dx, dy)


class AIAgent(Agent):
    """Agent controlled by AI/VLM through API calls."""
    
    def __init__(
        self,
        name: str = "AI Assistant",
        x: float = 700.0,
        y: float = 300.0,
        color: Tuple[int, int, int] = (50, 205, 50),  # Lime green
        **kwargs
    ):
        super().__init__(
            name=name,
            x=x,
            y=y,
            color=color,
            agent_type=AgentType.AI,
            **kwargs
        )
        self._target_x: Optional[float] = None
        self._target_y: Optional[float] = None
        self._message: str = ""  # Message to display above agent
    
    @property
    def message(self) -> str:
        return self._message
    
    def say(self, message: str) -> None:
        """Set a message to display above the AI agent."""
        self._message = message
    
    def clear_message(self) -> None:
        """Clear the displayed message."""
        self._message = ""
    
    def move_towards(self, target_x: float, target_y: float, arrival_threshold: float = 10.0) -> bool:
        """
        Move towards a target position.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            arrival_threshold: Distance at which we consider arrived
            
        Returns:
            True if arrived at target, False if still moving
        """
        self._target_x = target_x
        self._target_y = target_y
        
        distance = self.distance_to_point(target_x, target_y)
        
        if distance <= arrival_threshold:
            self.stop()
            self._target_x = None
            self._target_y = None
            return True
        
        # Calculate direction
        dx = (target_x - self.x) / distance
        dy = (target_y - self.y) / distance
        
        self.set_movement(dx, dy)
        return False
    
    def move_direction(self, direction: str, magnitude: float = 1.0) -> None:
        """
        Move in a cardinal direction.
        
        Args:
            direction: One of 'up', 'down', 'left', 'right', 
                      'up_left', 'up_right', 'down_left', 'down_right'
            magnitude: Movement magnitude (0.0 to 1.0)
        """
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-0.707, -0.707),
            'up_right': (0.707, -0.707),
            'down_left': (-0.707, 0.707),
            'down_right': (0.707, 0.707),
        }
        
        if direction in directions:
            dx, dy = directions[direction]
            self.set_movement(dx * magnitude, dy * magnitude)
        else:
            self.stop()
    
    def get_status(self) -> dict:
        """Get current AI agent status for debugging/display."""
        return {
            "name": self.name,
            "position": (round(self.x, 1), round(self.y, 1)),
            "velocity": (round(self.velocity_x, 2), round(self.velocity_y, 2)),
            "target": (self._target_x, self._target_y) if self._target_x else None,
            "message": self._message,
        }
