#!/usr/bin/env python3
"""
Phase 1 Test: PyMunk-based physics simulation.

Uses proper rigid body dynamics from PyMunk (same as push-T diffusion policy):
- PD control for smooth agent movement
- Proper collision response
- Multiple shape types (box, T, L)
- Two goal zones

Run with:
    cd /home/mani/Repos/proactive_hcdt
    conda activate ur5_python
    python tests/test_simulation_pymunk.py
"""

import sys
import math
import random

sys.path.insert(0, '/home/mani/Repos/proactive_hcdt')

from proactive_hcdt.simulation.physics import PhysicsWorld, PhysicsConfig, check_pymunk_available
from proactive_hcdt.simulation.pymunk_renderer import PymunkRenderer, GoalZone

# Check dependencies
if not check_pymunk_available():
    print("PyMunk not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymunk"])
    from proactive_hcdt.simulation.physics import PhysicsWorld, PhysicsConfig


def setup_game(physics: PhysicsWorld, goals: list) -> None:
    """Set up objects and goals for the game."""
    
    # Clear existing
    physics.clear_objects()
    goals.clear()
    
    width = physics.width
    height = physics.height
    
    # Create goal zones
    goals.append(GoalZone(
        name="Blue Zone",
        x=100,
        y=height / 2,
        width=120,
        height=180,
        assigned_to="human"
    ))
    
    goals.append(GoalZone(
        name="Green Zone",
        x=width - 100,
        y=height / 2,
        width=120,
        height=180,
        assigned_to="ai"
    ))
    
    # Add objects in center area
    center_x = width / 2
    center_y = height / 2
    
    # Boxes
    physics.add_box("Box-1", center_x - 60, center_y - 80, size=45,
                    color=(255, 165, 0))
    physics.add_box("Box-2", center_x + 80, center_y + 60, size=45,
                    color=(255, 140, 0))
    
    # T-shapes (like push-T)
    physics.add_tee("T-1", center_x - 30, center_y + 40, scale=22,
                    angle=random.uniform(-0.3, 0.3), color=(220, 20, 60))
    physics.add_tee("T-2", center_x + 50, center_y - 30, scale=22,
                    angle=random.uniform(-0.3, 0.3), color=(178, 34, 34))
    
    # L-shapes
    physics.add_ell("L-1", center_x + 20, center_y + 100, scale=18,
                    angle=random.uniform(-0.5, 0.5), color=(100, 149, 237))
    physics.add_ell("L-2", center_x - 80, center_y - 20, scale=18,
                    angle=random.uniform(-0.5, 0.5), color=(70, 130, 180))


class AIController:
    """Simple AI that pushes objects to its goal zone."""
    
    def __init__(self, physics: PhysicsWorld, goals: list, agent_name: str = "ai"):
        self.physics = physics
        self.goals = goals
        self.agent_name = agent_name
        self.target_object = None
        self.state = "idle"
        self.message = ""
        self.tick = 0
        self.last_message_tick = 0
    
    def get_ai_goal(self) -> GoalZone:
        for goal in self.goals:
            if goal.assigned_to == "ai":
                return goal
        return None
    
    def update(self) -> None:
        self.tick += 1
        
        ai_goal = self.get_ai_goal()
        if not ai_goal:
            return
        
        ai_pos = self.physics.get_agent_position(self.agent_name)
        human_pos = self.physics.get_agent_position("human")
        
        # Initial greeting
        if self.tick < 60:
            if self.tick == 30:
                self.message = "Hi! I'll help push to the green zone!"
            return
        
        # Find object to target
        if self.target_object is None or self.tick % 120 == 0:
            best_obj = None
            best_score = float('inf')
            
            for name, body in self.physics.objects.items():
                # Skip if in a goal
                in_goal = False
                for goal in self.goals:
                    if goal.contains_point(body.position.x, body.position.y):
                        in_goal = True
                        break
                if in_goal:
                    continue
                
                # Score: prefer objects far from human, close to AI
                dist_to_ai = math.sqrt(
                    (body.position.x - ai_pos[0])**2 + 
                    (body.position.y - ai_pos[1])**2
                )
                dist_to_human = math.sqrt(
                    (body.position.x - human_pos[0])**2 + 
                    (body.position.y - human_pos[1])**2
                )
                
                # Prefer objects human isn't near
                score = dist_to_ai - dist_to_human * 0.5
                
                if score < best_score:
                    best_score = score
                    best_obj = name
            
            if best_obj and best_obj != self.target_object:
                self.target_object = best_obj
                if self.tick - self.last_message_tick > 120:
                    self.message = f"I'll get {best_obj}!"
                    self.last_message_tick = self.tick
        
        # Move towards target
        if self.target_object and self.target_object in self.physics.objects:
            obj = self.physics.objects[self.target_object]
            
            # Check if reached goal
            if ai_goal.contains_point(obj.position.x, obj.position.y):
                self.target_object = None
                if self.tick - self.last_message_tick > 60:
                    self.message = "Got one! ✓"
                    self.last_message_tick = self.tick
                return
            
            # Position behind object relative to goal
            dx = ai_goal.x - obj.position.x
            dy = ai_goal.y - obj.position.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 1:
                # Push position: behind object, away from goal
                push_dist = 30  # How far behind object to position
                target_x = obj.position.x - (dx / dist) * push_dist
                target_y = obj.position.y - (dy / dist) * push_dist
                
                # Set AI target (PD controller will smoothly move there)
                self.physics.set_agent_target(self.agent_name, target_x, target_y)
        else:
            # No target, stay put
            self.physics.set_agent_target(self.agent_name, ai_pos[0], ai_pos[1])
        
        # Clear message after a while
        if self.message and self.tick - self.last_message_tick > 90:
            self.message = ""


def main():
    print("=" * 60)
    print("PyMunk Physics Test: Smooth Push Dynamics")
    print("=" * 60)
    print()
    print("Using PyMunk rigid body physics (like push-T environment)")
    print("  - PD control for smooth agent movement")
    print("  - Proper collision response")
    print("  - No jitter!")
    print()
    print("Controls:")
    print("  WASD / Arrow Keys: Move human agent (blue)")
    print("  R: Reset")
    print("  ESC: Quit")
    print()
    
    # Configuration
    width, height = 900, 650
    
    # Physics config (matching push-T)
    config = PhysicsConfig(
        sim_hz=100,
        control_hz=60,
        k_p=100.0,
        k_v=20.0,
        damping=0.0,
        friction=1.0,
    )
    
    # Create physics world
    physics = PhysicsWorld(width, height, config)
    
    # Add agents
    physics.add_agent("human", width * 0.2, height * 0.5, radius=18,
                      color=(65, 105, 225))  # Royal blue
    physics.add_agent("ai", width * 0.8, height * 0.5, radius=18,
                      color=(50, 205, 50))  # Lime green
    
    # Create renderer
    renderer = PymunkRenderer(width, height, "PyMunk Physics: Human-AI Collaboration")
    renderer.initialize()
    
    # Game state
    goals = []
    setup_game(physics, goals)
    
    # AI controller
    ai = AIController(physics, goals, "ai")
    
    # Score tracking
    human_score = 0
    ai_score = 0
    scored_objects = set()
    
    tick = 0
    running = True
    
    # Agent speed (for keyboard control)
    move_speed = 4.0
    
    print("Starting simulation...")
    
    try:
        while running:
            tick += 1
            
            # Process input
            keys = renderer.process_events()
            
            if keys['quit']:
                running = False
                continue
            
            if keys['r']:
                print("Resetting...")
                setup_game(physics, goals)
                ai = AIController(physics, goals, "ai")
                physics.reset_agents()
                # Reset agent positions
                physics.agents["human"].position = (width * 0.2, height * 0.5)
                physics.agents["ai"].position = (width * 0.8, height * 0.5)
                human_score = 0
                ai_score = 0
                scored_objects.clear()
                tick = 0
            
            # Human agent control
            human_pos = physics.get_agent_position("human")
            dx, dy = 0, 0
            
            if keys['w'] or keys['up']:
                dy -= move_speed
            if keys['s'] or keys['down']:
                dy += move_speed
            if keys['a'] or keys['left']:
                dx -= move_speed
            if keys['d'] or keys['right']:
                dx += move_speed
            
            # Normalize diagonal movement
            if dx != 0 and dy != 0:
                dx *= 0.707
                dy *= 0.707
            
            # Set human target (will be smoothly approached via PD control)
            target_x = human_pos[0] + dx * 10
            target_y = human_pos[1] + dy * 10
            
            # Clamp to world bounds
            target_x = max(30, min(width - 30, target_x))
            target_y = max(30, min(height - 30, target_y))
            
            physics.set_agent_target("human", target_x, target_y)
            
            # Update AI
            ai.update()
            
            # Step physics
            physics.step()
            
            # Check goals and scoring
            for goal in goals:
                for name, body in physics.objects.items():
                    was_inside = name in goal.objects_inside
                    is_inside = goal.contains_point(body.position.x, body.position.y)
                    
                    if is_inside:
                        goal.objects_inside.add(name)
                    else:
                        goal.objects_inside.discard(name)
                    
                    # Score on first entry
                    if is_inside and not was_inside and name not in scored_objects:
                        scored_objects.add(name)
                        if goal.assigned_to == "human":
                            human_score += 100
                            print(f"Human scored! ({name}) Total: {human_score}")
                        elif goal.assigned_to == "ai":
                            ai_score += 100
                            print(f"AI scored! ({name}) Total: {ai_score}")
            
            # Render
            renderer.render(
                physics.space,
                goals=goals,
                human_score=human_score,
                ai_score=ai_score,
                tick=tick,
            )
            
            # Draw agent labels and messages on top
            human_pos = physics.get_agent_position("human")
            ai_pos = physics.get_agent_position("ai")
            
            renderer.draw_label(human_pos[0], human_pos[1], "Human", offset_y=25)
            renderer.draw_label(ai_pos[0], ai_pos[1], "AI", offset_y=25)
            
            if ai.message:
                renderer.draw_speech_bubble(ai_pos[0], ai_pos[1], ai.message, radius=18)
            
            # Update display again for labels
            import pygame
            pygame.display.flip()
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        renderer.cleanup()
        print()
        print("=" * 40)
        print("Final Scores:")
        print(f"  Human: {human_score}")
        print(f"  AI:    {ai_score}")
        print("=" * 40)


if __name__ == "__main__":
    main()
