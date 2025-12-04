"""
Pygame-based renderer for the simulation environment.

Handles all visual output including agents, shapes, goals, and UI elements.
Supports rendering of Box, L-shape, and T-shape objects with rotation.
"""

from typing import Optional, TYPE_CHECKING, List, Tuple
import math

if TYPE_CHECKING:
    from .world import World, GoalZone
    from .agents import Agent, AIAgent
    from .shapes import PushableObject

# Pygame import with graceful fallback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None


class Renderer:
    """
    Pygame-based renderer for the simulation.
    
    Supports:
    - Multiple shape types (Box, L, T) with rotation
    - Goal zones with color coding per agent
    - Speech bubbles for AI communication
    - Score display per agent
    """
    
    def __init__(self, world: "World"):
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "Pygame is required for rendering. Install with: pip install pygame"
            )
        
        self.world = world
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.tiny_font: Optional[pygame.font.Font] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize pygame and create the window."""
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode(
            (self.world.width, self.world.height)
        )
        pygame.display.set_caption(self.world.config.title)
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 16)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
        
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
    
    def process_events(self) -> dict:
        """
        Process pygame events and return keyboard state.
        
        Returns:
            Dict with key states for movement controls
        """
        keys_pressed = {
            'w': False, 's': False, 'a': False, 'd': False,
            'up': False, 'down': False, 'left': False, 'right': False,
            'space': False, 'escape': False, 'r': False,
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.world.stop()
                return keys_pressed
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.world.stop()
                    return keys_pressed
                if event.key == pygame.K_r:
                    keys_pressed['r'] = True
        
        # Get continuous key state
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
    
    def render(self) -> float:
        """
        Render the current world state.
        
        Returns:
            Delta time in seconds for this frame
        """
        # Clear screen
        self.screen.fill(self.world.config.background_color)
        
        # Draw goal zones (behind everything)
        for goal in self.world.goals:
            self._draw_goal(goal)
        
        # Draw objects (shapes)
        for obj in self.world.objects:
            self._draw_shape(obj)
        
        # Draw agents
        self._draw_agent(self.world.human_agent, is_human=True)
        self._draw_agent(self.world.ai_agent, is_human=False)
        
        # Draw AI message if any
        if self.world.ai_agent.message:
            self._draw_speech_bubble(
                self.world.ai_agent,
                self.world.ai_agent.message
            )
        
        # Draw UI
        self._draw_ui()
        
        # Update display
        pygame.display.flip()
        
        # Tick clock and return dt
        dt = self.clock.tick(self.world.config.fps) / 1000.0
        return dt
    
    def _draw_agent(self, agent: "Agent", is_human: bool) -> None:
        """Draw an agent circle with label."""
        # Draw agent circle
        pygame.draw.circle(
            self.screen,
            agent.color,
            (int(agent.x), int(agent.y)),
            int(agent.radius)
        )
        
        # Draw outline
        outline_color = (255, 255, 255) if is_human else (0, 0, 0)
        pygame.draw.circle(
            self.screen,
            outline_color,
            (int(agent.x), int(agent.y)),
            int(agent.radius),
            3  # outline width
        )
        
        # Draw direction indicator if moving
        if abs(agent.velocity_x) > 0.1 or abs(agent.velocity_y) > 0.1:
            end_x = agent.x + agent.velocity_x * 5
            end_y = agent.y + agent.velocity_y * 5
            pygame.draw.line(
                self.screen,
                outline_color,
                (int(agent.x), int(agent.y)),
                (int(end_x), int(end_y)),
                2
            )
        
        # Draw label below agent
        label = self.small_font.render(agent.name, True, (50, 50, 50))
        label_rect = label.get_rect(
            center=(int(agent.x), int(agent.y + agent.radius + 15))
        )
        self.screen.blit(label, label_rect)
    
    def _draw_shape(self, obj: "PushableObject") -> None:
        """Draw a pushable shape with rotation."""
        from .shapes import ShapeType
        
        if obj.shape_type == ShapeType.BOX:
            self._draw_box(obj)
        elif obj.shape_type == ShapeType.L_SHAPE:
            self._draw_l_shape(obj)
        elif obj.shape_type == ShapeType.T_SHAPE:
            self._draw_t_shape(obj)
        else:
            self._draw_box(obj)  # Fallback
        
        # Draw label
        label = self.tiny_font.render(obj.name, True, (30, 30, 30))
        label_rect = label.get_rect(center=(int(obj.x), int(obj.y)))
        self.screen.blit(label, label_rect)
    
    def _rotate_point(self, px: float, py: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
        """Rotate a point around a center."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = px - cx
        dy = py - cy
        return (
            cx + dx * cos_a - dy * sin_a,
            cy + dx * sin_a + dy * cos_a
        )
    
    def _draw_rotated_rect(self, cx: float, cy: float, w: float, h: float, 
                           angle: float, color: Tuple[int, int, int], 
                           outline_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """Draw a rotated rectangle."""
        # Calculate corners
        hw, hh = w / 2, h / 2
        corners = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        
        # Rotate and translate corners
        rotated = []
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for x, y in corners:
            rx = cx + x * cos_a - y * sin_a
            ry = cy + x * sin_a + y * cos_a
            rotated.append((rx, ry))
        
        # Draw filled polygon
        pygame.draw.polygon(self.screen, color, rotated)
        pygame.draw.polygon(self.screen, outline_color, rotated, 2)
    
    def _draw_box(self, obj: "PushableObject") -> None:
        """Draw a simple box."""
        self._draw_rotated_rect(
            obj.x, obj.y, obj.size, obj.size,
            obj.rotation, obj.color
        )
    
    def _draw_l_shape(self, obj: "PushableObject") -> None:
        """Draw an L-shaped object."""
        # L-shape consists of two rectangles
        bar_width = obj.size * 0.4
        
        # Vertical bar (left side)
        v_cx = obj.x + (-obj.size * 0.3) * math.cos(obj.rotation) - (-obj.size * 0.3) * math.sin(obj.rotation)
        v_cy = obj.y + (-obj.size * 0.3) * math.sin(obj.rotation) + (-obj.size * 0.3) * math.cos(obj.rotation)
        self._draw_rotated_rect(v_cx, v_cy, bar_width, obj.size * 1.4, obj.rotation, obj.color)
        
        # Horizontal bar (bottom)
        h_cx = obj.x + (obj.size * 0.2) * math.cos(obj.rotation) - (obj.size * 0.4) * math.sin(obj.rotation)
        h_cy = obj.y + (obj.size * 0.2) * math.sin(obj.rotation) + (obj.size * 0.4) * math.cos(obj.rotation)
        self._draw_rotated_rect(h_cx, h_cy, obj.size * 0.8, bar_width, obj.rotation, obj.color)
    
    def _draw_t_shape(self, obj: "PushableObject") -> None:
        """Draw a T-shaped object."""
        bar_width = obj.size * 0.4
        
        # Horizontal bar (top)
        h_cx = obj.x + 0 * math.cos(obj.rotation) - (-obj.size * 0.4) * math.sin(obj.rotation)
        h_cy = obj.y + 0 * math.sin(obj.rotation) + (-obj.size * 0.4) * math.cos(obj.rotation)
        self._draw_rotated_rect(h_cx, h_cy, obj.size * 1.2, bar_width, obj.rotation, obj.color)
        
        # Vertical bar (center stem)
        v_cx = obj.x + 0 * math.cos(obj.rotation) - (obj.size * 0.2) * math.sin(obj.rotation)
        v_cy = obj.y + 0 * math.sin(obj.rotation) + (obj.size * 0.2) * math.cos(obj.rotation)
        self._draw_rotated_rect(v_cx, v_cy, bar_width, obj.size * 0.8, obj.rotation, obj.color)
    
    def _draw_goal(self, goal: "GoalZone") -> None:
        """Draw a goal zone rectangle with agent color coding."""
        rect = pygame.Rect(
            goal.x - goal.width / 2,
            goal.y - goal.height / 2,
            goal.width,
            goal.height
        )
        
        # Color based on assignment
        if goal.assigned_to == "human":
            fill_color = (200, 220, 255)  # Light blue for human
            border_color = (65, 105, 225)  # Royal blue
        elif goal.assigned_to == "ai":
            fill_color = (200, 255, 200)  # Light green for AI
            border_color = (50, 205, 50)  # Lime green
        else:
            fill_color = (255, 255, 200)  # Light yellow for shared
            border_color = (200, 200, 0)
        
        # Draw filled with border
        pygame.draw.rect(self.screen, fill_color, rect)
        pygame.draw.rect(self.screen, border_color, rect, 4)
        
        # Draw dashed inner border to indicate goal area
        dash_length = 10
        for i in range(0, int(goal.width), dash_length * 2):
            # Top edge
            pygame.draw.line(self.screen, border_color,
                           (goal.x - goal.width/2 + i, goal.y - goal.height/2 + 8),
                           (goal.x - goal.width/2 + i + dash_length, goal.y - goal.height/2 + 8), 2)
            # Bottom edge
            pygame.draw.line(self.screen, border_color,
                           (goal.x - goal.width/2 + i, goal.y + goal.height/2 - 8),
                           (goal.x - goal.width/2 + i + dash_length, goal.y + goal.height/2 - 8), 2)
        
        # Draw label
        label_text = goal.name
        if goal.assigned_to:
            label_text += f" [{goal.assigned_to.upper()}]"
        label = self.small_font.render(label_text, True, border_color)
        label_rect = label.get_rect(center=(int(goal.x), int(goal.y - goal.height/2 - 15)))
        self.screen.blit(label, label_rect)
        
        # Show objects inside count
        if goal.objects_inside:
            count_text = f"{len(goal.objects_inside)} inside"
            count_label = self.tiny_font.render(count_text, True, border_color)
            count_rect = count_label.get_rect(center=(int(goal.x), int(goal.y + goal.height/2 + 12)))
            self.screen.blit(count_label, count_rect)
    
    def _draw_speech_bubble(self, agent: "AIAgent", message: str) -> None:
        """Draw a speech bubble above an agent."""
        padding = 10
        max_width = 200
        
        # Word wrap the message
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
        
        # Calculate dimensions
        line_height = self.small_font.get_linesize()
        bubble_height = len(lines) * line_height + 2 * padding
        bubble_width = max(self.small_font.size(line)[0] for line in lines) + 2 * padding
        
        # Position bubble above agent
        bubble_x = agent.x - bubble_width / 2
        bubble_y = agent.y - agent.radius - bubble_height - 20
        
        # Clamp to screen
        bubble_x = max(5, min(self.world.width - bubble_width - 5, bubble_x))
        bubble_y = max(5, bubble_y)
        
        # Draw bubble background
        bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_width, bubble_height)
        pygame.draw.rect(self.screen, (255, 255, 255), bubble_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 100), bubble_rect, 2, border_radius=8)
        
        # Draw triangle pointer
        pointer_points = [
            (agent.x - 8, bubble_y + bubble_height),
            (agent.x + 8, bubble_y + bubble_height),
            (agent.x, bubble_y + bubble_height + 10)
        ]
        pygame.draw.polygon(self.screen, (255, 255, 255), pointer_points)
        pygame.draw.lines(self.screen, (100, 100, 100), False, pointer_points[1:], 2)
        
        # Draw text
        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, (30, 30, 30))
            text_rect = text.get_rect(
                topleft=(bubble_x + padding, bubble_y + padding + i * line_height)
            )
            self.screen.blit(text, text_rect)
    
    def _draw_ui(self) -> None:
        """Draw UI overlay (scores, instructions, etc.)."""
        # Human score (left side, blue)
        human_score_text = self.font.render(f"Human: {self.world.human_score}", True, (65, 105, 225))
        self.screen.blit(human_score_text, (10, 10))
        
        # AI score (right side, green)  
        ai_score_text = self.font.render(f"AI: {self.world.ai_score}", True, (50, 205, 50))
        ai_rect = ai_score_text.get_rect(topright=(self.world.width - 10, 10))
        self.screen.blit(ai_score_text, ai_rect)
        
        # Tick counter (center)
        tick_text = self.small_font.render(f"Tick: {self.world.tick}", True, (100, 100, 100))
        tick_rect = tick_text.get_rect(midtop=(self.world.width // 2, 10))
        self.screen.blit(tick_text, tick_rect)
        
        # Time elapsed
        state = self.world.get_state()
        time_text = self.small_font.render(f"Time: {state.time_elapsed:.1f}s", True, (100, 100, 100))
        time_rect = time_text.get_rect(midtop=(self.world.width // 2, 30))
        self.screen.blit(time_text, time_rect)
        
        # Instructions
        instructions = [
            "WASD/Arrows: Move Human (blue)",
            "AI (green) will assist automatically",
            "Push shapes to matching goal zones!",
            "R: Reset | ESC: Quit",
        ]
        y = self.world.height - 18 * len(instructions) - 10
        for instruction in instructions:
            text = self.tiny_font.render(instruction, True, (120, 120, 120))
            self.screen.blit(text, (10, y))
            y += 18
