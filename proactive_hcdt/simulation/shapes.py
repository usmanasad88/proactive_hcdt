"""
Shape definitions for pushable objects.

Supports various shapes: Box, L-shape, T-shape with proper collision geometry.
Uses compliant contact dynamics similar to the push-T environment.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import math


class ShapeType(Enum):
    BOX = "box"
    L_SHAPE = "L"
    T_SHAPE = "T"


@dataclass
class CollisionRect:
    """A rectangular collision primitive (relative to parent center)."""
    offset_x: float  # Offset from parent center
    offset_y: float
    width: float
    height: float


def get_shape_collision_rects(shape_type: ShapeType, size: float = 40.0) -> List[CollisionRect]:
    """
    Get collision rectangles for a shape type.
    
    Args:
        shape_type: Type of shape
        size: Base unit size
        
    Returns:
        List of CollisionRect relative to shape center
    """
    if shape_type == ShapeType.BOX:
        return [CollisionRect(0, 0, size, size)]
    
    elif shape_type == ShapeType.L_SHAPE:
        # L-shape: vertical bar + horizontal bar at bottom
        #   █
        #   █
        #   █ █ █
        bar_width = size * 0.4
        return [
            # Vertical bar (left side)
            CollisionRect(-size * 0.3, -size * 0.3, bar_width, size * 1.4),
            # Horizontal bar (bottom)
            CollisionRect(size * 0.2, size * 0.4, size * 0.8, bar_width),
        ]
    
    elif shape_type == ShapeType.T_SHAPE:
        # T-shape: horizontal bar on top + vertical bar
        #   █ █ █
        #     █
        #     █
        bar_width = size * 0.4
        return [
            # Horizontal bar (top)
            CollisionRect(0, -size * 0.4, size * 1.2, bar_width),
            # Vertical bar (center)
            CollisionRect(0, size * 0.2, bar_width, size * 0.8),
        ]
    
    return [CollisionRect(0, 0, size, size)]


@dataclass
class PushableObject:
    """
    A pushable object with shape geometry and compliant physics.
    
    Uses quasi-static pushing model similar to push-T:
    - Objects move only when pushed (no momentum/inertia flying)
    - Compliant contact with smooth position updates
    - Rotation support for complex shapes
    """
    
    name: str
    x: float
    y: float
    shape_type: ShapeType = ShapeType.BOX
    size: float = 40.0
    rotation: float = 0.0  # Radians
    color: Tuple[int, int, int] = (255, 165, 0)  # Orange
    
    # Physics parameters (compliant pushing)
    pushable: bool = True
    push_damping: float = 0.85  # How quickly velocity decays (higher = more damping)
    push_compliance: float = 0.4  # How easily object moves when pushed (0-1)
    rotation_compliance: float = 0.02  # How easily object rotates
    
    # State
    velocity_x: float = field(default=0.0, repr=False)
    velocity_y: float = field(default=0.0, repr=False)
    angular_velocity: float = field(default=0.0, repr=False)
    
    # Cached collision geometry
    _collision_rects: List[CollisionRect] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        self._collision_rects = get_shape_collision_rects(self.shape_type, self.size)
    
    def get_world_collision_rects(self) -> List[Tuple[float, float, float, float, float]]:
        """
        Get collision rectangles in world coordinates.
        
        Returns:
            List of (center_x, center_y, width, height, rotation) tuples
        """
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        
        result = []
        for rect in self._collision_rects:
            # Rotate offset
            world_x = self.x + rect.offset_x * cos_r - rect.offset_y * sin_r
            world_y = self.y + rect.offset_x * sin_r + rect.offset_y * cos_r
            result.append((world_x, world_y, rect.width, rect.height, self.rotation))
        
        return result
    
    def get_bounding_radius(self) -> float:
        """Get approximate bounding radius for broad-phase collision."""
        if self.shape_type == ShapeType.BOX:
            return self.size * 0.71  # sqrt(2)/2
        else:
            return self.size * 1.2  # Larger for complex shapes
    
    def apply_push(self, push_x: float, push_y: float, contact_x: float, contact_y: float) -> None:
        """
        Apply a push force with compliant dynamics.
        
        Args:
            push_x, push_y: Push direction (normalized)
            contact_x, contact_y: Contact point in world coords
        """
        # Linear push (compliant - directly affects velocity, not acceleration)
        self.velocity_x += push_x * self.push_compliance
        self.velocity_y += push_y * self.push_compliance
        
        # Calculate torque from off-center pushes
        dx = contact_x - self.x
        dy = contact_y - self.y
        
        # Cross product for torque (2D: dx * push_y - dy * push_x)
        torque = dx * push_y - dy * push_x
        self.angular_velocity += torque * self.rotation_compliance
    
    def update(self, dt: float, world_width: float, world_height: float) -> None:
        """Update object with compliant damping."""
        # Apply velocity with damping
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.rotation += self.angular_velocity
        
        # Apply damping (quasi-static: rapid decay)
        self.velocity_x *= (1.0 - self.push_damping)
        self.velocity_y *= (1.0 - self.push_damping)
        self.angular_velocity *= (1.0 - self.push_damping)
        
        # Stop if very slow
        if abs(self.velocity_x) < 0.001:
            self.velocity_x = 0
        if abs(self.velocity_y) < 0.001:
            self.velocity_y = 0
        if abs(self.angular_velocity) < 0.0001:
            self.angular_velocity = 0
        
        # Keep rotation in [-pi, pi]
        while self.rotation > math.pi:
            self.rotation -= 2 * math.pi
        while self.rotation < -math.pi:
            self.rotation += 2 * math.pi
        
        # Clamp to world bounds (using bounding radius)
        radius = self.get_bounding_radius()
        self.x = max(radius, min(world_width - radius, self.x))
        self.y = max(radius, min(world_height - radius, self.y))
    
    def point_inside(self, px: float, py: float) -> bool:
        """Check if a point is inside any collision rect."""
        for wx, wy, w, h, rot in self.get_world_collision_rects():
            # Transform point to rect's local space
            dx = px - wx
            dy = py - wy
            cos_r = math.cos(-rot)
            sin_r = math.sin(-rot)
            local_x = dx * cos_r - dy * sin_r
            local_y = dx * sin_r + dy * cos_r
            
            if abs(local_x) <= w / 2 and abs(local_y) <= h / 2:
                return True
        return False
    
    def get_closest_point_on_shape(self, px: float, py: float) -> Tuple[float, float, float]:
        """
        Get the closest point on the shape surface to a given point.
        
        Returns:
            (closest_x, closest_y, distance)
        """
        best_dist = float('inf')
        best_point = (self.x, self.y)
        
        for wx, wy, w, h, rot in self.get_world_collision_rects():
            # Transform point to rect's local space
            dx = px - wx
            dy = py - wy
            cos_r = math.cos(-rot)
            sin_r = math.sin(-rot)
            local_x = dx * cos_r - dy * sin_r
            local_y = dx * sin_r + dy * cos_r
            
            # Clamp to rect bounds
            clamped_x = max(-w / 2, min(w / 2, local_x))
            clamped_y = max(-h / 2, min(h / 2, local_y))
            
            # Transform back to world
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            world_closest_x = wx + clamped_x * cos_r - clamped_y * sin_r
            world_closest_y = wy + clamped_x * sin_r + clamped_y * cos_r
            
            dist = math.sqrt((px - world_closest_x) ** 2 + (py - world_closest_y) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_point = (world_closest_x, world_closest_y)
        
        return (*best_point, best_dist)


def create_box(name: str, x: float, y: float, size: float = 40.0, 
               color: Tuple[int, int, int] = (255, 165, 0)) -> PushableObject:
    """Create a simple box object."""
    return PushableObject(
        name=name, x=x, y=y, 
        shape_type=ShapeType.BOX, 
        size=size, 
        color=color
    )


def create_l_shape(name: str, x: float, y: float, size: float = 50.0,
                   color: Tuple[int, int, int] = (100, 149, 237),  # Cornflower blue
                   rotation: float = 0.0) -> PushableObject:
    """Create an L-shaped object."""
    return PushableObject(
        name=name, x=x, y=y,
        shape_type=ShapeType.L_SHAPE,
        size=size,
        color=color,
        rotation=rotation
    )


def create_t_shape(name: str, x: float, y: float, size: float = 50.0,
                   color: Tuple[int, int, int] = (220, 20, 60),  # Crimson
                   rotation: float = 0.0) -> PushableObject:
    """Create a T-shaped object."""
    return PushableObject(
        name=name, x=x, y=y,
        shape_type=ShapeType.T_SHAPE,
        size=size,
        color=color,
        rotation=rotation
    )
