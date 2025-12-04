"""
World simulation environment.

The World manages the simulation state, including agents, objects, and goals.
Uses compliant contact dynamics similar to the push-T environment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
import time
import math

from .agents import Agent, HumanAgent, AIAgent, AgentState
from .shapes import PushableObject, ShapeType, create_box, create_l_shape, create_t_shape


@dataclass 
class GoalZone:
    """A target zone where objects should be pushed to."""
    
    name: str
    x: float
    y: float
    width: float = 100.0
    height: float = 100.0
    color: Tuple[int, int, int] = (144, 238, 144)  # Light green
    assigned_to: str = ""  # "human", "ai", or "" for shared
    target_shapes: List[ShapeType] = field(default_factory=list)  # Which shapes belong here
    
    # Track which objects are inside
    objects_inside: Set[str] = field(default_factory=set)
    
    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is inside the goal zone."""
        return (
            self.x - self.width / 2 <= px <= self.x + self.width / 2 and
            self.y - self.height / 2 <= py <= self.y + self.height / 2
        )
    
    def check_object(self, obj: PushableObject) -> bool:
        """Check if object center is in zone and update tracking."""
        inside = self.contains_point(obj.x, obj.y)
        if inside:
            self.objects_inside.add(obj.name)
        else:
            self.objects_inside.discard(obj.name)
        return inside


@dataclass
class WorldConfig:
    """Configuration for the simulation world."""
    
    width: int = 1000
    height: int = 700
    background_color: Tuple[int, int, int] = (245, 245, 245)
    fps: int = 60
    title: str = "Human-AI Collaboration: Push Objects to Goals"
    
    # Physics
    agent_push_strength: float = 2.5  # How hard agents push objects
    collision_separation: float = 0.5  # How much to separate on collision


@dataclass
class WorldState:
    """Immutable snapshot of world state for observation."""
    
    tick: int
    time_elapsed: float
    human_agent: AgentState
    ai_agent: AgentState
    objects: List[Dict[str, Any]]
    goals: List[Dict[str, Any]]
    human_score: int
    ai_score: int
    
    def to_description(self) -> str:
        """Convert world state to natural language description for VLM."""
        lines = [
            f"=== World State (tick {self.tick}, {self.time_elapsed:.1f}s) ===",
            f"Human Score: {self.human_score} | AI Score: {self.ai_score}",
            "",
            f"Human Agent '{self.human_agent.name}':",
            f"  Position: ({self.human_agent.x:.0f}, {self.human_agent.y:.0f})",
            f"  Moving: {'Yes' if abs(self.human_agent.velocity_x) > 0.1 or abs(self.human_agent.velocity_y) > 0.1 else 'No'}",
            "",
            f"AI Agent '{self.ai_agent.name}':",
            f"  Position: ({self.ai_agent.x:.0f}, {self.ai_agent.y:.0f})",
            "",
        ]
        
        if self.objects:
            lines.append("Objects in play:")
            for obj in self.objects:
                status = f"in {obj.get('in_goal', 'play')}" if obj.get('in_goal') else "in play"
                lines.append(f"  - {obj['name']} ({obj['shape']}): ({obj['x']:.0f}, {obj['y']:.0f}) - {status}")
        
        if self.goals:
            lines.append("\nGoal Zones:")
            for goal in self.goals:
                assigned = f"[{goal['assigned_to']}]" if goal['assigned_to'] else "[shared]"
                objects_in = goal.get('objects_inside', [])
                lines.append(f"  - {goal['name']} {assigned}: ({goal['x']:.0f}, {goal['y']:.0f})")
                if objects_in:
                    lines.append(f"    Contains: {', '.join(objects_in)}")
        
        return "\n".join(lines)


class World:
    """
    Main simulation world with compliant push physics.
    
    Features:
    - Multiple pushable shapes (Box, L, T)
    - Two goal zones (one per agent)
    - Compliant contact dynamics (objects don't fly away)
    - Score tracking per agent
    """
    
    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        
        # Create agents
        self.human_agent = HumanAgent(
            x=self.config.width * 0.15,
            y=self.config.height * 0.5
        )
        self.ai_agent = AIAgent(
            x=self.config.width * 0.85,
            y=self.config.height * 0.5
        )
        
        # World contents
        self.objects: List[PushableObject] = []
        self.goals: List[GoalZone] = []
        
        # Score tracking
        self._human_score = 0
        self._ai_score = 0
        self._scored_objects: Set[str] = set()  # Track which objects have scored
        
        # Simulation state
        self._tick = 0
        self._start_time = time.time()
        self._running = True
    
    @property
    def width(self) -> int:
        return self.config.width
    
    @property
    def height(self) -> int:
        return self.config.height
    
    @property
    def tick(self) -> int:
        return self._tick
    
    @property
    def human_score(self) -> int:
        return self._human_score
    
    @property
    def ai_score(self) -> int:
        return self._ai_score
    
    @property
    def score(self) -> int:
        """Combined score for backward compatibility."""
        return self._human_score + self._ai_score
    
    @property
    def running(self) -> bool:
        return self._running
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
    
    def add_object(self, obj: PushableObject) -> None:
        """Add a pushable object to the world."""
        self.objects.append(obj)
    
    def add_goal(self, goal: GoalZone) -> None:
        """Add a goal zone to the world."""
        self.goals.append(goal)
    
    def update(self, dt: float) -> None:
        """Update world simulation by one step."""
        self._tick += 1
        
        # Update agents
        self.human_agent.update(dt, self.width, self.height)
        self.ai_agent.update(dt, self.width, self.height)
        
        # Handle agent-object collisions with compliant pushing
        self._handle_agent_object_collisions(self.human_agent)
        self._handle_agent_object_collisions(self.ai_agent)
        
        # Handle object-object collisions
        self._handle_object_object_collisions()
        
        # Update objects
        for obj in self.objects:
            obj.update(dt, self.width, self.height)
        
        # Check goals and update scores
        self._check_goals()
    
    def _handle_agent_object_collisions(self, agent: Agent) -> None:
        """Handle compliant collisions between an agent and objects."""
        for obj in self.objects:
            if not obj.pushable:
                continue
            
            # Get closest point on object to agent center
            closest_x, closest_y, dist = obj.get_closest_point_on_shape(agent.x, agent.y)
            
            # Check for collision
            if dist < agent.radius:
                # Calculate push direction (from agent to object)
                if dist > 0.01:
                    push_x = (closest_x - agent.x) / dist
                    push_y = (closest_y - agent.y) / dist
                else:
                    # Agent center inside object, push away from object center
                    dx = obj.x - agent.x
                    dy = obj.y - agent.y
                    d = math.sqrt(dx * dx + dy * dy)
                    if d > 0.01:
                        push_x = -dx / d
                        push_y = -dy / d
                    else:
                        push_x, push_y = 1.0, 0.0
                
                # Apply compliant push
                push_strength = self.config.agent_push_strength
                
                # Stronger push if agent is actively moving into object
                agent_vel_mag = math.sqrt(agent.velocity_x**2 + agent.velocity_y**2)
                if agent_vel_mag > 0.1:
                    # Check if moving towards object
                    vel_dot = (agent.velocity_x * push_x + agent.velocity_y * push_y)
                    if vel_dot > 0:
                        push_strength *= (1.0 + vel_dot * 0.3)
                
                obj.apply_push(
                    push_x * push_strength,
                    push_y * push_strength,
                    closest_x,
                    closest_y
                )
                
                # Slight separation to prevent overlap
                overlap = agent.radius - dist
                if overlap > 0:
                    sep = self.config.collision_separation
                    obj.x += push_x * overlap * sep
                    obj.y += push_y * overlap * sep
    
    def _handle_object_object_collisions(self) -> None:
        """Handle collisions between objects (simple separation)."""
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i + 1:]:
                # Simple bounding circle check
                dx = obj2.x - obj1.x
                dy = obj2.y - obj1.y
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = obj1.get_bounding_radius() + obj2.get_bounding_radius()
                
                if dist < min_dist * 0.8:  # Allow some overlap since shapes aren't circles
                    # Separate objects
                    if dist > 0.01:
                        nx = dx / dist
                        ny = dy / dist
                    else:
                        nx, ny = 1.0, 0.0
                    
                    overlap = min_dist * 0.8 - dist
                    sep = overlap * 0.3
                    
                    obj1.x -= nx * sep
                    obj1.y -= ny * sep
                    obj2.x += nx * sep
                    obj2.y += ny * sep
    
    def _check_goals(self) -> None:
        """Check if objects are in goal zones and update scores."""
        for goal in self.goals:
            for obj in self.objects:
                was_inside = obj.name in goal.objects_inside
                is_inside = goal.check_object(obj)
                
                # Score when object enters goal (first time only)
                if is_inside and not was_inside and obj.name not in self._scored_objects:
                    # Check if this goal accepts this shape type
                    shape_match = (not goal.target_shapes or 
                                   obj.shape_type in goal.target_shapes)
                    
                    if shape_match:
                        self._scored_objects.add(obj.name)
                        points = 100
                        
                        if goal.assigned_to == "human":
                            self._human_score += points
                        elif goal.assigned_to == "ai":
                            self._ai_score += points
                        else:
                            # Shared goal - split points
                            self._human_score += points // 2
                            self._ai_score += points // 2
    
    def get_object_by_name(self, name: str) -> Optional[PushableObject]:
        """Get an object by name."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None
    
    def get_state(self) -> WorldState:
        """Get current world state snapshot."""
        # Determine which goal each object is in
        object_goals = {}
        for goal in self.goals:
            for obj_name in goal.objects_inside:
                object_goals[obj_name] = goal.name
        
        return WorldState(
            tick=self._tick,
            time_elapsed=time.time() - self._start_time,
            human_agent=self.human_agent.get_state(),
            ai_agent=self.ai_agent.get_state(),
            objects=[
                {
                    "name": obj.name,
                    "x": obj.x,
                    "y": obj.y,
                    "shape": obj.shape_type.value,
                    "rotation": obj.rotation,
                    "size": obj.size,
                    "in_goal": object_goals.get(obj.name),
                }
                for obj in self.objects
            ],
            goals=[
                {
                    "name": goal.name,
                    "x": goal.x,
                    "y": goal.y,
                    "width": goal.width,
                    "height": goal.height,
                    "assigned_to": goal.assigned_to,
                    "objects_inside": list(goal.objects_inside),
                }
                for goal in self.goals
            ],
            human_score=self._human_score,
            ai_score=self._ai_score,
        )
    
    def get_observation_for_ai(self) -> str:
        """Get a text description of the world for the AI agent."""
        return self.get_state().to_description()
    
    def reset(self) -> None:
        """Reset the world to initial state."""
        self.human_agent.x = self.config.width * 0.15
        self.human_agent.y = self.config.height * 0.5
        self.human_agent.stop()
        
        self.ai_agent.x = self.config.width * 0.85
        self.ai_agent.y = self.config.height * 0.5
        self.ai_agent.stop()
        self.ai_agent.clear_message()
        
        self.objects.clear()
        self.goals.clear()
        
        self._human_score = 0
        self._ai_score = 0
        self._scored_objects.clear()
        
        self._tick = 0
        self._start_time = time.time()
        self._running = True


# Keep backward compatibility alias
GameObject = PushableObject
