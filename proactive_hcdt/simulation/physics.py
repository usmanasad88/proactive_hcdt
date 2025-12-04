"""
PyMunk-based physics simulation for the collaborative environment.

Uses the same physics approach as the push-T diffusion policy environment:
- PyMunk for rigid body dynamics
- PD control for agent movement (smooth, not jerky)
- Proper collision handling with friction
- Kinematic agents pushing dynamic objects

This provides much smoother and more realistic pushing behavior.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math

try:
    import pymunk
    from pymunk.vec2d import Vec2d
    PYMUNK_AVAILABLE = True
except ImportError:
    PYMUNK_AVAILABLE = False
    pymunk = None
    Vec2d = None


class ShapeType(Enum):
    BOX = "box"
    L_SHAPE = "L"
    T_SHAPE = "T"


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation."""
    
    # Simulation parameters
    sim_hz: int = 100  # Physics steps per second
    control_hz: int = 60  # Control/render rate
    
    # Agent PD control gains (from push-T)
    k_p: float = 100.0  # Proportional gain
    k_v: float = 20.0   # Derivative (velocity) gain
    
    # Physics properties
    damping: float = 0.0  # Space damping (0 = no global damping)
    friction: float = 1.0  # Surface friction
    
    # Object mass
    block_mass: float = 1.0


class PhysicsWorld:
    """
    PyMunk-based physics world matching push-T environment.
    
    Key features:
    - Kinematic agents with PD control (smooth movement)
    - Dynamic pushable objects
    - Proper collision response
    - Wall boundaries
    """
    
    def __init__(self, width: int, height: int, config: Optional[PhysicsConfig] = None):
        if not PYMUNK_AVAILABLE:
            raise ImportError(
                "PyMunk is required for physics. Install with: pip install pymunk"
            )
        
        self.width = width
        self.height = height
        self.config = config or PhysicsConfig()
        
        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # Top-down view, no gravity
        self.space.damping = self.config.damping
        
        # Track bodies
        self.agents: Dict[str, pymunk.Body] = {}
        self.objects: Dict[str, pymunk.Body] = {}
        self.agent_targets: Dict[str, Vec2d] = {}  # Target positions for PD control
        
        # Collision tracking
        self.n_contact_points = 0
        
        # Add walls
        self._add_walls()
        
        # Setup collision handler (API changed in pymunk 7.x)
        # We just track contacts without needing the handler for basic simulation
        # The physics works fine without explicit collision tracking
    
    def _add_walls(self) -> None:
        """Add boundary walls to the physics space."""
        wall_thickness = 5
        walls = [
            # (start, end)
            ((wall_thickness, self.height - wall_thickness), 
             (wall_thickness, wall_thickness)),  # Left
            ((wall_thickness, wall_thickness), 
             (self.width - wall_thickness, wall_thickness)),  # Top
            ((self.width - wall_thickness, wall_thickness), 
             (self.width - wall_thickness, self.height - wall_thickness)),  # Right
            ((wall_thickness, self.height - wall_thickness), 
             (self.width - wall_thickness, self.height - wall_thickness)),  # Bottom
        ]
        
        for start, end in walls:
            shape = pymunk.Segment(self.space.static_body, start, end, wall_thickness)
            shape.friction = self.config.friction
            shape.color = (200, 200, 200, 255)  # Light gray
            self.space.add(shape)
    
    def _handle_collision(self, arbiter, space, data) -> None:
        """Track collision contact points."""
        self.n_contact_points += len(arbiter.contact_point_set.points)
    
    def add_agent(self, name: str, x: float, y: float, radius: float = 15.0,
                  color: Tuple[int, int, int] = (65, 105, 225)) -> pymunk.Body:
        """
        Add a kinematic agent (controlled via PD control).
        
        Kinematic bodies don't respond to forces but can push dynamic objects.
        """
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (x, y)
        
        shape = pymunk.Circle(body, radius)
        shape.friction = self.config.friction
        shape.color = (*color, 255)
        
        self.space.add(body, shape)
        self.agents[name] = body
        self.agent_targets[name] = Vec2d(x, y)
        
        return body
    
    def add_box(self, name: str, x: float, y: float, size: float = 40.0,
                color: Tuple[int, int, int] = (255, 165, 0)) -> pymunk.Body:
        """Add a dynamic box that can be pushed."""
        mass = self.config.block_mass
        inertia = pymunk.moment_for_box(mass, (size, size))
        
        body = pymunk.Body(mass, inertia)
        body.position = (x, y)
        body.friction = self.config.friction
        
        shape = pymunk.Poly.create_box(body, (size, size))
        shape.friction = self.config.friction
        shape.color = (*color, 255)
        
        self.space.add(body, shape)
        self.objects[name] = body
        
        return body
    
    def add_tee(self, name: str, x: float, y: float, scale: float = 30.0,
                angle: float = 0.0, color: Tuple[int, int, int] = (220, 20, 60)) -> pymunk.Body:
        """
        Add a T-shaped object (from push-T environment).
        
        The T consists of two rectangles sharing a body.
        """
        mass = self.config.block_mass
        length = 4  # Ratio
        
        # Vertices for horizontal bar (top of T)
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0)
        ]
        
        # Vertices for vertical bar (stem of T)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale)
        ]
        
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        
        body = pymunk.Body(mass, inertia1 + inertia2)
        
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        
        shape1.friction = self.config.friction
        shape2.friction = self.config.friction
        shape1.color = (*color, 255)
        shape2.color = (*color, 255)
        
        # Set center of gravity to average of both shapes
        body.center_of_gravity = (
            (shape1.center_of_gravity[0] + shape2.center_of_gravity[0]) / 2,
            (shape1.center_of_gravity[1] + shape2.center_of_gravity[1]) / 2
        )
        
        body.position = (x, y)
        body.angle = angle
        
        self.space.add(body, shape1, shape2)
        self.objects[name] = body
        
        return body
    
    def add_ell(self, name: str, x: float, y: float, scale: float = 30.0,
                angle: float = 0.0, color: Tuple[int, int, int] = (100, 149, 237)) -> pymunk.Body:
        """
        Add an L-shaped object.
        
        The L consists of two rectangles sharing a body.
        """
        mass = self.config.block_mass
        length = 3  # Ratio
        
        # Vertices for vertical bar (tall part of L)
        vertices1 = [
            (-scale / 2, -length * scale / 2),
            (scale / 2, -length * scale / 2),
            (scale / 2, length * scale / 2),
            (-scale / 2, length * scale / 2)
        ]
        
        # Vertices for horizontal bar (bottom of L)
        vertices2 = [
            (scale / 2, length * scale / 2 - scale),
            (length * scale / 2 + scale / 2, length * scale / 2 - scale),
            (length * scale / 2 + scale / 2, length * scale / 2),
            (scale / 2, length * scale / 2)
        ]
        
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        
        body = pymunk.Body(mass, inertia1 + inertia2)
        
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        
        shape1.friction = self.config.friction
        shape2.friction = self.config.friction
        shape1.color = (*color, 255)
        shape2.color = (*color, 255)
        
        # Set center of gravity
        body.center_of_gravity = (
            (shape1.center_of_gravity[0] + shape2.center_of_gravity[0]) / 2,
            (shape1.center_of_gravity[1] + shape2.center_of_gravity[1]) / 2
        )
        
        body.position = (x, y)
        body.angle = angle
        
        self.space.add(body, shape1, shape2)
        self.objects[name] = body
        
        return body
    
    def set_agent_target(self, name: str, target_x: float, target_y: float) -> None:
        """Set the target position for an agent (for PD control)."""
        if name in self.agent_targets:
            self.agent_targets[name] = Vec2d(target_x, target_y)
    
    def get_agent_position(self, name: str) -> Tuple[float, float]:
        """Get current agent position."""
        if name in self.agents:
            pos = self.agents[name].position
            return (pos.x, pos.y)
        return (0, 0)
    
    def get_agent_velocity(self, name: str) -> Tuple[float, float]:
        """Get current agent velocity."""
        if name in self.agents:
            vel = self.agents[name].velocity
            return (vel.x, vel.y)
        return (0, 0)
    
    def get_object_state(self, name: str) -> Dict[str, Any]:
        """Get object position, angle, and velocity."""
        if name in self.objects:
            body = self.objects[name]
            return {
                "x": body.position.x,
                "y": body.position.y,
                "angle": body.angle,
                "velocity": (body.velocity.x, body.velocity.y),
                "angular_velocity": body.angular_velocity,
            }
        return {}
    
    def step(self, dt: float = None) -> None:
        """
        Step the physics simulation with PD control for agents.
        
        This matches the push-T environment approach:
        - Multiple physics sub-steps per control step
        - PD control for smooth agent movement
        """
        if dt is None:
            dt = 1.0 / self.config.control_hz
        
        physics_dt = 1.0 / self.config.sim_hz
        n_steps = max(1, int(self.config.sim_hz / self.config.control_hz))
        
        self.n_contact_points = 0
        
        for _ in range(n_steps):
            # Apply PD control to each agent
            for name, body in self.agents.items():
                target = self.agent_targets.get(name, body.position)
                
                # PD control: acceleration = k_p * (target - pos) + k_v * (0 - vel)
                acceleration = (
                    self.config.k_p * (target - body.position) +
                    self.config.k_v * (Vec2d(0, 0) - body.velocity)
                )
                
                # Update velocity (integrate acceleration)
                body.velocity += acceleration * physics_dt
            
            # Step physics
            self.space.step(physics_dt)
    
    def clear_objects(self) -> None:
        """Remove all dynamic objects from the simulation."""
        for name, body in list(self.objects.items()):
            for shape in body.shapes:
                self.space.remove(shape)
            self.space.remove(body)
        self.objects.clear()
    
    def reset_agents(self) -> None:
        """Stop all agents and clear their targets."""
        for name, body in self.agents.items():
            body.velocity = (0, 0)
            self.agent_targets[name] = body.position


def check_pymunk_available() -> bool:
    """Check if PyMunk is available."""
    return PYMUNK_AVAILABLE
