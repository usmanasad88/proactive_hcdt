"""
AI Action Primitives for the collaborative simulation.

Provides high-level actions that an AI/VLM can invoke:
- Perception: observe world state, find objects, check goals
- Movement: move to position, approach object, push object to goal
- Communication: display messages to human

These primitives bridge the gap between natural language AI reasoning
and low-level physics simulation control.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math
import time


class ActionStatus(Enum):
    """Status of an action execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    status: ActionStatus
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectInfo:
    """Information about an object in the world."""
    name: str
    shape_type: str  # "box", "T", "L"
    x: float
    y: float
    angle: float
    in_goal: Optional[str] = None  # Name of goal zone if inside one
    distance_to_human: float = 0.0
    distance_to_ai: float = 0.0


@dataclass
class GoalInfo:
    """Information about a goal zone."""
    name: str
    x: float
    y: float
    width: float
    height: float
    assigned_to: str  # "human", "ai", or ""
    objects_inside: List[str] = field(default_factory=list)


@dataclass 
class WorldObservation:
    """Complete observation of the world state."""
    tick: int
    human_position: Tuple[float, float]
    human_velocity: Tuple[float, float]
    ai_position: Tuple[float, float]
    ai_velocity: Tuple[float, float]
    objects: List[ObjectInfo]
    goals: List[GoalInfo]
    human_score: int
    ai_score: int
    
    def to_text(self) -> str:
        """Convert to natural language description for VLM."""
        lines = [
            f"=== World State (tick {self.tick}) ===",
            f"Scores - Human: {self.human_score}, AI: {self.ai_score}",
            "",
            f"Human Agent: position ({self.human_position[0]:.0f}, {self.human_position[1]:.0f})",
            f"AI Agent (you): position ({self.ai_position[0]:.0f}, {self.ai_position[1]:.0f})",
            "",
            "Objects:",
        ]
        
        for obj in self.objects:
            status = f"in {obj.in_goal}" if obj.in_goal else "in play"
            lines.append(
                f"  - {obj.name} ({obj.shape_type}): ({obj.x:.0f}, {obj.y:.0f}), "
                f"dist to human: {obj.distance_to_human:.0f}, dist to AI: {obj.distance_to_ai:.0f}, "
                f"status: {status}"
            )
        
        lines.append("")
        lines.append("Goal Zones:")
        for goal in self.goals:
            owner = f"[{goal.assigned_to}]" if goal.assigned_to else "[shared]"
            inside = f", contains: {', '.join(goal.objects_inside)}" if goal.objects_inside else ""
            lines.append(f"  - {goal.name} {owner}: center ({goal.x:.0f}, {goal.y:.0f}){inside}")
        
        return "\n".join(lines)
    
    def get_objects_in_play(self) -> List[ObjectInfo]:
        """Get objects not yet in any goal."""
        return [obj for obj in self.objects if obj.in_goal is None]
    
    def get_closest_object_to_ai(self) -> Optional[ObjectInfo]:
        """Get the closest object to the AI that's still in play."""
        in_play = self.get_objects_in_play()
        if not in_play:
            return None
        return min(in_play, key=lambda o: o.distance_to_ai)
    
    def get_object_by_name(self, name: str) -> Optional[ObjectInfo]:
        """Find object by name."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None
    
    def get_ai_goal(self) -> Optional[GoalInfo]:
        """Get the AI's assigned goal zone."""
        for goal in self.goals:
            if goal.assigned_to == "ai":
                return goal
        return None


class AIActionPrimitives:
    """
    High-level action primitives for AI control.
    
    Provides structured actions that can be called by a VLM:
    - observe(): Get current world state
    - move_to(x, y): Move AI agent to position
    - push_object_to_goal(object_name, goal_name): Push an object to a goal
    - say(message): Display a message
    
    Actions are non-blocking and return immediately. Use observe() to
    check progress.
    """
    
    def __init__(self, physics_world, goals: List, agent_name: str = "ai"):
        """
        Initialize with physics world and goals.
        
        Args:
            physics_world: PhysicsWorld instance
            goals: List of GoalZone instances
            agent_name: Name of the AI agent in physics world
        """
        self.physics = physics_world
        self.goals = goals
        self.agent_name = agent_name
        
        # Current action state
        self._current_action: Optional[str] = None
        self._action_target: Optional[Dict[str, Any]] = None
        self._action_start_tick: int = 0
        self._message: str = ""
        
        # Action parameters
        self.arrival_threshold = 25.0  # Distance to consider "arrived"
        self.push_offset = 35.0  # Distance behind object when pushing
        self.action_timeout = 600  # Max ticks for an action
    
    @property
    def message(self) -> str:
        """Current message to display."""
        return self._message
    
    def clear_message(self) -> None:
        """Clear the displayed message."""
        self._message = ""
    
    # ==================== PERCEPTION ====================
    
    def observe(self) -> WorldObservation:
        """
        Get complete observation of the current world state.
        
        Returns:
            WorldObservation with all entities and their states
        """
        human_pos = self.physics.get_agent_position("human")
        human_vel = self.physics.get_agent_velocity("human")
        ai_pos = self.physics.get_agent_position(self.agent_name)
        ai_vel = self.physics.get_agent_velocity(self.agent_name)
        
        # Build object info
        objects = []
        for name, body in self.physics.objects.items():
            # Determine shape type from name
            if name.startswith("Box"):
                shape_type = "box"
            elif name.startswith("T-"):
                shape_type = "T"
            elif name.startswith("L-"):
                shape_type = "L"
            else:
                shape_type = "unknown"
            
            # Check if in any goal
            in_goal = None
            for goal in self.goals:
                if goal.contains_point(body.position.x, body.position.y):
                    in_goal = goal.name
                    break
            
            # Calculate distances
            dist_human = math.sqrt(
                (body.position.x - human_pos[0])**2 +
                (body.position.y - human_pos[1])**2
            )
            dist_ai = math.sqrt(
                (body.position.x - ai_pos[0])**2 +
                (body.position.y - ai_pos[1])**2
            )
            
            objects.append(ObjectInfo(
                name=name,
                shape_type=shape_type,
                x=body.position.x,
                y=body.position.y,
                angle=body.angle,
                in_goal=in_goal,
                distance_to_human=dist_human,
                distance_to_ai=dist_ai,
            ))
        
        # Build goal info
        goal_infos = []
        for goal in self.goals:
            # Check which objects are inside
            objects_inside = []
            for name, body in self.physics.objects.items():
                if goal.contains_point(body.position.x, body.position.y):
                    objects_inside.append(name)
            
            goal_infos.append(GoalInfo(
                name=goal.name,
                x=goal.x,
                y=goal.y,
                width=goal.width,
                height=goal.height,
                assigned_to=goal.assigned_to,
                objects_inside=objects_inside,
            ))
        
        # Calculate scores (simplified - count objects * 100)
        human_score = 0
        ai_score = 0
        for goal_info in goal_infos:
            if goal_info.assigned_to == "human":
                human_score = len(goal_info.objects_inside) * 100
            elif goal_info.assigned_to == "ai":
                ai_score = len(goal_info.objects_inside) * 100
        
        return WorldObservation(
            tick=0,  # Would need to track this
            human_position=human_pos,
            human_velocity=human_vel,
            ai_position=ai_pos,
            ai_velocity=ai_vel,
            objects=objects,
            goals=goal_infos,
            human_score=human_score,
            ai_score=ai_score,
        )
    
    def find_objects_by_type(self, shape_type: str) -> List[ObjectInfo]:
        """
        Find all objects of a given type.
        
        Args:
            shape_type: "box", "T", or "L"
            
        Returns:
            List of matching objects
        """
        obs = self.observe()
        return [o for o in obs.objects if o.shape_type.lower() == shape_type.lower()]
    
    def find_unscored_objects(self) -> List[ObjectInfo]:
        """Find objects not yet in any goal zone."""
        obs = self.observe()
        return obs.get_objects_in_play()
    
    def get_object_position(self, object_name: str) -> Optional[Tuple[float, float]]:
        """Get position of a specific object."""
        if object_name in self.physics.objects:
            body = self.physics.objects[object_name]
            return (body.position.x, body.position.y)
        return None
    
    def get_goal_position(self, goal_name: str) -> Optional[Tuple[float, float]]:
        """Get center position of a goal zone."""
        for goal in self.goals:
            if goal.name == goal_name:
                return (goal.x, goal.y)
        return None
    
    # ==================== MOVEMENT ACTIONS ====================
    
    def move_to(self, x: float, y: float) -> ActionResult:
        """
        Move AI agent to a specific position.
        
        Args:
            x, y: Target position
            
        Returns:
            ActionResult indicating if movement started
        """
        # Clamp to world bounds
        x = max(30, min(self.physics.width - 30, x))
        y = max(30, min(self.physics.height - 30, y))
        
        self.physics.set_agent_target(self.agent_name, x, y)
        
        self._current_action = "move_to"
        self._action_target = {"x": x, "y": y}
        
        return ActionResult(
            success=True,
            status=ActionStatus.IN_PROGRESS,
            message=f"Moving to ({x:.0f}, {y:.0f})",
            data={"target_x": x, "target_y": y}
        )
    
    def move_to_object(self, object_name: str, offset_direction: str = "behind") -> ActionResult:
        """
        Move AI agent near an object.
        
        Args:
            object_name: Name of object to approach
            offset_direction: "behind" (relative to AI goal), "left", "right", "above", "below"
            
        Returns:
            ActionResult
        """
        obj_pos = self.get_object_position(object_name)
        if obj_pos is None:
            return ActionResult(
                success=False,
                status=ActionStatus.FAILED,
                message=f"Object '{object_name}' not found"
            )
        
        # Calculate offset position
        offset = self.push_offset
        
        if offset_direction == "behind":
            # Position behind object relative to AI's goal
            ai_goal = None
            for goal in self.goals:
                if goal.assigned_to == "ai":
                    ai_goal = goal
                    break
            
            if ai_goal:
                dx = ai_goal.x - obj_pos[0]
                dy = ai_goal.y - obj_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    target_x = obj_pos[0] - (dx / dist) * offset
                    target_y = obj_pos[1] - (dy / dist) * offset
                else:
                    target_x, target_y = obj_pos[0] - offset, obj_pos[1]
            else:
                target_x, target_y = obj_pos[0] - offset, obj_pos[1]
        elif offset_direction == "left":
            target_x, target_y = obj_pos[0] - offset, obj_pos[1]
        elif offset_direction == "right":
            target_x, target_y = obj_pos[0] + offset, obj_pos[1]
        elif offset_direction == "above":
            target_x, target_y = obj_pos[0], obj_pos[1] - offset
        elif offset_direction == "below":
            target_x, target_y = obj_pos[0], obj_pos[1] + offset
        else:
            target_x, target_y = obj_pos
        
        return self.move_to(target_x, target_y)
    
    def push_towards(self, object_name: str, target_x: float, target_y: float) -> ActionResult:
        """
        Push an object towards a target position.
        
        The AI positions itself behind the object and moves through it.
        
        Args:
            object_name: Object to push
            target_x, target_y: Where to push it
            
        Returns:
            ActionResult
        """
        obj_pos = self.get_object_position(object_name)
        if obj_pos is None:
            return ActionResult(
                success=False,
                status=ActionStatus.FAILED,
                message=f"Object '{object_name}' not found"
            )
        
        # Move towards the object (which will push it towards target)
        # The AI should position itself on the opposite side of the object from the target
        dx = target_x - obj_pos[0]
        dy = target_y - obj_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 10:
            return ActionResult(
                success=True,
                status=ActionStatus.COMPLETED,
                message=f"Object '{object_name}' is already at target"
            )
        
        # Push by moving through the object towards target
        # Target position is slightly past the object towards goal
        push_target_x = obj_pos[0] + (dx / dist) * 20
        push_target_y = obj_pos[1] + (dy / dist) * 20
        
        self.physics.set_agent_target(self.agent_name, push_target_x, push_target_y)
        
        self._current_action = "push_towards"
        self._action_target = {
            "object": object_name,
            "target_x": target_x,
            "target_y": target_y
        }
        
        return ActionResult(
            success=True,
            status=ActionStatus.IN_PROGRESS,
            message=f"Pushing {object_name} towards ({target_x:.0f}, {target_y:.0f})"
        )
    
    def push_object_to_goal(self, object_name: str, goal_name: str) -> ActionResult:
        """
        Push an object to a specific goal zone.
        
        This is a high-level action that handles positioning and pushing.
        
        Args:
            object_name: Name of object to push (e.g., "Box-1", "T-1", "L-2")
            goal_name: Name of goal zone (e.g., "Green Zone", "Blue Zone")
            
        Returns:
            ActionResult
        """
        # Validate object exists
        obj_pos = self.get_object_position(object_name)
        if obj_pos is None:
            return ActionResult(
                success=False,
                status=ActionStatus.FAILED,
                message=f"Object '{object_name}' not found"
            )
        
        # Validate goal exists
        goal_pos = self.get_goal_position(goal_name)
        if goal_pos is None:
            return ActionResult(
                success=False,
                status=ActionStatus.FAILED,
                message=f"Goal '{goal_name}' not found"
            )
        
        # Check if already in goal
        for goal in self.goals:
            if goal.name == goal_name:
                if goal.contains_point(obj_pos[0], obj_pos[1]):
                    return ActionResult(
                        success=True,
                        status=ActionStatus.COMPLETED,
                        message=f"Object '{object_name}' is already in '{goal_name}'"
                    )
        
        # Start pushing
        self._current_action = "push_to_goal"
        self._action_target = {
            "object": object_name,
            "goal": goal_name,
            "goal_x": goal_pos[0],
            "goal_y": goal_pos[1]
        }
        
        self._message = f"Pushing {object_name} to {goal_name}"
        
        return self.push_towards(object_name, goal_pos[0], goal_pos[1])
    
    # ==================== COMMUNICATION ====================
    
    def say(self, message: str) -> ActionResult:
        """
        Display a message in a speech bubble.
        
        Args:
            message: Text to display
            
        Returns:
            ActionResult
        """
        self._message = message
        return ActionResult(
            success=True,
            status=ActionStatus.COMPLETED,
            message=f"Saying: {message}"
        )
    
    # ==================== ACTION MANAGEMENT ====================
    
    def get_current_action(self) -> Optional[str]:
        """Get the name of the current action being executed."""
        return self._current_action
    
    def is_action_complete(self) -> bool:
        """Check if the current action has completed."""
        if self._current_action is None:
            return True
        
        ai_pos = self.physics.get_agent_position(self.agent_name)
        
        if self._current_action == "move_to":
            target = self._action_target
            dist = math.sqrt(
                (ai_pos[0] - target["x"])**2 +
                (ai_pos[1] - target["y"])**2
            )
            if dist < self.arrival_threshold:
                self._current_action = None
                return True
        
        elif self._current_action in ("push_towards", "push_to_goal"):
            target = self._action_target
            obj_pos = self.get_object_position(target.get("object", ""))
            
            if obj_pos:
                # Check if object reached target
                target_x = target.get("target_x") or target.get("goal_x", 0)
                target_y = target.get("target_y") or target.get("goal_y", 0)
                
                dist = math.sqrt(
                    (obj_pos[0] - target_x)**2 +
                    (obj_pos[1] - target_y)**2
                )
                
                if dist < 50:  # Object close to target
                    self._current_action = None
                    self._message = "Done! ✓"
                    return True
                
                # Continue pushing - update target to keep pushing
                self.push_towards(target["object"], target_x, target_y)
        
        return False
    
    def stop(self) -> ActionResult:
        """Stop the current action and stop moving."""
        ai_pos = self.physics.get_agent_position(self.agent_name)
        self.physics.set_agent_target(self.agent_name, ai_pos[0], ai_pos[1])
        
        self._current_action = None
        self._action_target = None
        
        return ActionResult(
            success=True,
            status=ActionStatus.COMPLETED,
            message="Stopped"
        )
    
    # ==================== STRATEGIC HELPERS ====================
    
    def select_best_object_to_push(self) -> Optional[str]:
        """
        Select the best object for AI to push based on strategy.
        
        Strategy:
        - Pick objects not in any goal
        - Prefer objects far from human (don't compete)
        - Prefer objects closer to AI (efficiency)
        
        Returns:
            Object name or None if no good targets
        """
        obs = self.observe()
        in_play = obs.get_objects_in_play()
        
        if not in_play:
            return None
        
        # Score each object
        best_obj = None
        best_score = float('-inf')
        
        for obj in in_play:
            # Higher score = better target
            # Prefer objects far from human (less competition)
            # Prefer objects close to AI (less travel)
            score = obj.distance_to_human * 0.5 - obj.distance_to_ai
            
            if score > best_score:
                best_score = score
                best_obj = obj.name
        
        return best_obj
    
    def execute_autonomous_step(self) -> Optional[ActionResult]:
        """
        Execute one step of autonomous behavior.
        
        This can be called each frame to make the AI act on its own.
        
        Returns:
            ActionResult if an action was taken, None if idle
        """
        # If we have an ongoing action, continue it
        if self._current_action and not self.is_action_complete():
            return None  # Still executing
        
        # Find something to do
        obs = self.observe()
        ai_goal = obs.get_ai_goal()
        
        if not ai_goal:
            return None
        
        # Find best object to push
        target_obj = self.select_best_object_to_push()
        
        if target_obj:
            return self.push_object_to_goal(target_obj, ai_goal.name)
        else:
            # All objects scored
            if not self._message:
                self.say("All done! 🎉")
            return None
