"""
PyMunk-integrated renderer for the simulation environment.

Uses PyMunk's debug drawing utilities for accurate shape rendering,
matching the push-T diffusion policy environment style.
"""

from typing import Optional, TYPE_CHECKING, List, Tuple, Dict, Any
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

try:
    import pymunk
    import pymunk.pygame_util
    from pymunk.space_debug_draw_options import SpaceDebugColor
    from pymunk.vec2d import Vec2d
    PYMUNK_AVAILABLE = True
except ImportError:
    PYMUNK_AVAILABLE = False
    pymunk = None


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convert pymunk coordinates to pygame surface coordinates."""
    return (round(p[0]), round(p[1]))


def light_color(color: "SpaceDebugColor") -> "SpaceDebugColor":
    """Create a lighter version of a color for highlights."""
    import numpy as np
    c = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    return SpaceDebugColor(r=c[0], g=c[1], b=c[2], a=c[3])


class PymunkDrawOptions(pymunk.SpaceDebugDrawOptions):
    """
    Custom draw options for PyMunk that match push-T style.
    
    Draws shapes with nice fills and highlights.
    """
    
    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super().__init__()
    
    def draw_circle(
        self,
        pos: "Vec2d",
        angle: float,
        radius: float,
        outline_color: "SpaceDebugColor",
        fill_color: "SpaceDebugColor",
    ) -> None:
        p = to_pygame(pos, self.surface)
        
        # Fill
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        # Highlight (inner lighter circle)
        if radius > 8:
            pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0)
        
        # Direction indicator
        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        pygame.draw.line(self.surface, outline_color.as_int(), p, p2, line_r)
    
    def draw_segment(self, a: "Vec2d", b: "Vec2d", color: "SpaceDebugColor") -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])
    
    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: "SpaceDebugColor",
        fill_color: "SpaceDebugColor",
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(self.surface, fill_color.as_int(), p1, round(radius))
            pygame.draw.circle(self.surface, fill_color.as_int(), p2, round(radius))
    
    def draw_polygon(
        self,
        verts: List[Tuple[float, float]],
        radius: float,
        outline_color: "SpaceDebugColor",
        fill_color: "SpaceDebugColor",
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps_closed = ps + [ps[0]]
        
        # Draw filled polygon with highlight
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)
        
        # Draw edges with rounded corners
        edge_radius = 2
        if edge_radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, edge_radius, fill_color, fill_color)
    
    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: "SpaceDebugColor"
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


class GoalZone:
    """Visual goal zone (not physics-based)."""
    
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        width: float = 100.0,
        height: float = 100.0,
        color: Tuple[int, int, int] = (144, 238, 144),
        assigned_to: str = "",
    ):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.assigned_to = assigned_to
        self.objects_inside: set = set()
    
    def contains_point(self, px: float, py: float) -> bool:
        return (
            self.x - self.width / 2 <= px <= self.x + self.width / 2 and
            self.y - self.height / 2 <= py <= self.y + self.height / 2
        )


class PymunkRenderer:
    """
    Renderer that uses PyMunk's debug drawing for accurate physics visualization.
    """
    
    def __init__(self, width: int, height: int, title: str = "Simulation"):
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required. Install with: pip install pygame")
        if not PYMUNK_AVAILABLE:
            raise ImportError("PyMunk is required. Install with: pip install pymunk")
        
        self.width = width
        self.height = height
        self.title = title
        
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.draw_options: Optional[PymunkDrawOptions] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.tiny_font: Optional[pygame.font.Font] = None
        
        self._initialized = False
        self.fps = 60
        self.background_color = (245, 245, 245)
    
    def initialize(self) -> None:
        """Initialize pygame and create the window."""
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        
        self.clock = pygame.time.Clock()
        self.draw_options = PymunkDrawOptions(self.screen)
        
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 16)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
        
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
    
    def process_events(self) -> Dict[str, bool]:
        """Process pygame events and return keyboard state."""
        keys_pressed = {
            'w': False, 's': False, 'a': False, 'd': False,
            'up': False, 'down': False, 'left': False, 'right': False,
            'space': False, 'escape': False, 'r': False, 'quit': False,
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keys_pressed['quit'] = True
                return keys_pressed
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keys_pressed['quit'] = True
                    return keys_pressed
                if event.key == pygame.K_r:
                    keys_pressed['r'] = True
        
        # Continuous key state
        pressed = pygame.key.get_pressed()
        keys_pressed['w'] = pressed[pygame.K_w]
        keys_pressed['s'] = pressed[pygame.K_s]
        keys_pressed['a'] = pressed[pygame.K_a]
        keys_pressed['d'] = pressed[pygame.K_d]
        keys_pressed['up'] = pressed[pygame.K_UP]
        keys_pressed['down'] = pressed[pygame.K_DOWN]
        keys_pressed['left'] = pressed[pygame.K_LEFT]
        keys_pressed['right'] = pressed[pygame.K_RIGHT]
        keys_pressed['space'] = pressed[pygame.K_SPACE]
        
        return keys_pressed
    
    def render(
        self,
        physics_space: "pymunk.Space",
        goals: List[GoalZone] = None,
        human_score: int = 0,
        ai_score: int = 0,
        tick: int = 0,
        messages: Dict[str, str] = None,
        agent_labels: Dict[str, str] = None,
    ) -> float:
        """
        Render the physics space and UI.
        
        Returns:
            Delta time in seconds
        """
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw goal zones first (behind everything)
        if goals:
            for goal in goals:
                self._draw_goal(goal)
        
        # Draw physics objects using PyMunk debug draw
        physics_space.debug_draw(self.draw_options)
        
        # Draw agent labels
        if agent_labels:
            for name, label in agent_labels.items():
                # Find agent body position - we'll need to pass positions separately
                pass
        
        # Draw messages (speech bubbles)
        if messages:
            for agent_name, message in messages.items():
                if message:
                    # Would need agent position - skip for now or pass separately
                    pass
        
        # Draw UI
        self._draw_ui(human_score, ai_score, tick)
        
        # Update display
        pygame.display.flip()
        
        # Tick and return dt
        return self.clock.tick(self.fps) / 1000.0
    
    def _draw_goal(self, goal: GoalZone) -> None:
        """Draw a goal zone."""
        rect = pygame.Rect(
            goal.x - goal.width / 2,
            goal.y - goal.height / 2,
            goal.width,
            goal.height
        )
        
        # Color based on assignment
        if goal.assigned_to == "human":
            fill_color = (200, 220, 255)
            border_color = (65, 105, 225)
        elif goal.assigned_to == "ai":
            fill_color = (200, 255, 200)
            border_color = (50, 205, 50)
        else:
            fill_color = (255, 255, 200)
            border_color = (200, 200, 0)
        
        pygame.draw.rect(self.screen, fill_color, rect)
        pygame.draw.rect(self.screen, border_color, rect, 4)
        
        # Label
        label_text = goal.name
        if goal.assigned_to:
            label_text += f" [{goal.assigned_to.upper()}]"
        label = self.small_font.render(label_text, True, border_color)
        label_rect = label.get_rect(center=(int(goal.x), int(goal.y - goal.height/2 - 15)))
        self.screen.blit(label, label_rect)
    
    def _draw_ui(self, human_score: int, ai_score: int, tick: int) -> None:
        """Draw UI overlay."""
        # Human score (blue, left)
        human_text = self.font.render(f"Human: {human_score}", True, (65, 105, 225))
        self.screen.blit(human_text, (10, 10))
        
        # AI score (green, right)
        ai_text = self.font.render(f"AI: {ai_score}", True, (50, 205, 50))
        ai_rect = ai_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(ai_text, ai_rect)
        
        # Tick counter
        tick_text = self.small_font.render(f"Tick: {tick}", True, (100, 100, 100))
        tick_rect = tick_text.get_rect(midtop=(self.width // 2, 10))
        self.screen.blit(tick_text, tick_rect)
        
        # Instructions
        instructions = [
            "WASD/Arrows: Move Human (blue)",
            "AI (green) assists automatically",
            "Push shapes to goal zones!",
            "R: Reset | ESC: Quit",
        ]
        y = self.height - 18 * len(instructions) - 10
        for instruction in instructions:
            text = self.tiny_font.render(instruction, True, (120, 120, 120))
            self.screen.blit(text, (10, y))
            y += 18
    
    def draw_speech_bubble(self, x: float, y: float, message: str, radius: float = 15) -> None:
        """Draw a speech bubble at a position."""
        if not message:
            return
        
        padding = 8
        max_width = 180
        
        # Word wrap
        words = message.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if self.small_font.size(test_line)[0] <= max_width - 2 * padding:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        if not lines:
            return
        
        # Calculate bubble dimensions
        line_height = self.small_font.get_linesize()
        bubble_height = len(lines) * line_height + 2 * padding
        bubble_width = max(self.small_font.size(line)[0] for line in lines) + 2 * padding
        
        bubble_x = x - bubble_width / 2
        bubble_y = y - radius - bubble_height - 15
        
        # Clamp to screen
        bubble_x = max(5, min(self.width - bubble_width - 5, bubble_x))
        bubble_y = max(5, bubble_y)
        
        # Draw bubble
        bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_width, bubble_height)
        pygame.draw.rect(self.screen, (255, 255, 255), bubble_rect, border_radius=6)
        pygame.draw.rect(self.screen, (100, 100, 100), bubble_rect, 2, border_radius=6)
        
        # Draw pointer
        pointer_points = [
            (x - 6, bubble_y + bubble_height),
            (x + 6, bubble_y + bubble_height),
            (x, bubble_y + bubble_height + 8)
        ]
        pygame.draw.polygon(self.screen, (255, 255, 255), pointer_points)
        
        # Draw text
        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, (30, 30, 30))
            text_rect = text.get_rect(
                topleft=(bubble_x + padding, bubble_y + padding + i * line_height)
            )
            self.screen.blit(text, text_rect)
    
    def draw_label(self, x: float, y: float, text: str, offset_y: float = 20) -> None:
        """Draw a label below a position."""
        label = self.small_font.render(text, True, (50, 50, 50))
        label_rect = label.get_rect(center=(int(x), int(y + offset_y)))
        self.screen.blit(label, label_rect)
