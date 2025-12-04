#!/usr/bin/env python3
"""
Phase 2 Test: AI Action Primitives with PyMunk Physics.

Demonstrates the AI action primitive API:
- observe(): Get world state
- move_to(x, y): Move to position
- push_object_to_goal(object, goal): Push object to goal
- say(message): Display message

The AI uses these primitives to autonomously assist.

Run with:
    cd /home/mani/Repos/proactive_hcdt
    conda activate ur5_python
    python tests/test_ai_primitives.py
"""

import sys
import math
import random

sys.path.insert(0, '/home/mani/Repos/proactive_hcdt')

from proactive_hcdt.simulation.physics import PhysicsWorld, PhysicsConfig
from proactive_hcdt.simulation.pymunk_renderer import PymunkRenderer, GoalZone
from proactive_hcdt.simulation.ai_actions import AIActionPrimitives


def setup_game(physics: PhysicsWorld, goals: list) -> None:
    """Set up objects and goals."""
    physics.clear_objects()
    goals.clear()
    
    width = physics.width
    height = physics.height
    
    # Goal zones
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
    
    # Objects in center
    center_x = width / 2
    center_y = height / 2
    
    # Boxes
    physics.add_box("Box-1", center_x - 60, center_y - 80, size=45,
                    color=(255, 165, 0))
    physics.add_box("Box-2", center_x + 80, center_y + 60, size=45,
                    color=(255, 140, 0))
    
    # T-shapes
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
    """
    AI controller using the action primitives API.
    
    Demonstrates how a VLM could use the primitives to control the AI.
    """
    
    def __init__(self, primitives: AIActionPrimitives):
        self.ai = primitives
        self.tick = 0
        self.state = "greeting"
        self.current_target = None
        self.last_observation = None
    
    def update(self) -> None:
        """Update AI behavior using primitives."""
        self.tick += 1
        
        # Get fresh observation
        obs = self.ai.observe()
        self.last_observation = obs
        
        if self.state == "greeting":
            if self.tick == 30:
                self.ai.say("Hi! Using action primitives now!")
            elif self.tick == 120:
                self.ai.say("Let me find an object to push...")
                self.state = "selecting"
        
        elif self.state == "selecting":
            # Use the primitive to select best object
            target = self.ai.select_best_object_to_push()
            
            if target:
                self.current_target = target
                self.ai.say(f"I'll push {target} to Green Zone!")
                
                # Use the high-level push primitive
                result = self.ai.push_object_to_goal(target, "Green Zone")
                print(f"[AI] Action: push_object_to_goal('{target}', 'Green Zone') -> {result.message}")
                
                self.state = "pushing"
            else:
                self.ai.say("No more objects to push! 🎉")
                self.state = "done"
        
        elif self.state == "pushing":
            # Check if action completed
            if self.ai.is_action_complete():
                obj_info = obs.get_object_by_name(self.current_target)
                
                if obj_info and obj_info.in_goal == "Green Zone":
                    print(f"[AI] Successfully pushed {self.current_target} to goal!")
                    self.ai.say(f"Got {self.current_target}! ✓")
                    self.current_target = None
                    self.state = "cooldown"
                else:
                    # Keep pushing
                    if self.current_target:
                        self.ai.push_object_to_goal(self.current_target, "Green Zone")
            
            # Timeout - try a different object
            if self.tick % 400 == 0:
                self.ai.say("Hmm, trying a different approach...")
                self.current_target = None
                self.state = "selecting"
        
        elif self.state == "cooldown":
            # Brief pause before next action
            if self.tick % 60 == 0:
                self.ai.clear_message()
                self.state = "selecting"
        
        elif self.state == "done":
            pass  # Nothing more to do
    
    def print_observation(self) -> None:
        """Print current observation (for debugging/demo)."""
        if self.last_observation:
            print(self.last_observation.to_text())


def main():
    print("=" * 60)
    print("Phase 2: AI Action Primitives Demo")
    print("=" * 60)
    print()
    print("AI uses structured primitives:")
    print("  - ai.observe() -> WorldObservation")
    print("  - ai.move_to(x, y)")
    print("  - ai.push_object_to_goal(object_name, goal_name)")
    print("  - ai.say(message)")
    print()
    print("Controls:")
    print("  WASD/Arrows: Move human")
    print("  O: Print AI observation")
    print("  R: Reset")
    print("  ESC: Quit")
    print()
    
    # Setup
    width, height = 900, 650
    config = PhysicsConfig(sim_hz=100, control_hz=60, k_p=100.0, k_v=20.0)
    physics = PhysicsWorld(width, height, config)
    
    # Agents
    physics.add_agent("human", width * 0.2, height * 0.5, radius=18,
                      color=(65, 105, 225))
    physics.add_agent("ai", width * 0.8, height * 0.5, radius=18,
                      color=(50, 205, 50))
    
    # Renderer
    renderer = PymunkRenderer(width, height, "Phase 2: AI Action Primitives")
    renderer.initialize()
    
    # Game state
    goals = []
    setup_game(physics, goals)
    
    # AI with primitives
    ai_primitives = AIActionPrimitives(physics, goals, "ai")
    ai_controller = AIController(ai_primitives)
    
    # Scores
    human_score = 0
    ai_score = 0
    scored_objects = set()
    
    tick = 0
    running = True
    move_speed = 4.0
    
    print("Starting simulation...")
    print()
    
    # Print initial observation
    print("Initial world state:")
    print(ai_primitives.observe().to_text())
    print()
    
    try:
        while running:
            tick += 1
            
            # Input
            keys = renderer.process_events()
            
            if keys['quit']:
                running = False
                continue
            
            if keys['r']:
                print("\n[System] Resetting...")
                setup_game(physics, goals)
                physics.reset_agents()
                physics.agents["human"].position = (width * 0.2, height * 0.5)
                physics.agents["ai"].position = (width * 0.8, height * 0.5)
                ai_primitives = AIActionPrimitives(physics, goals, "ai")
                ai_controller = AIController(ai_primitives)
                human_score = 0
                ai_score = 0
                scored_objects.clear()
                tick = 0
            
            # 'O' key to print observation
            import pygame
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_o] and tick % 30 == 0:
                print("\n[Debug] Current observation:")
                ai_controller.print_observation()
                print()
            
            # Human control
            human_pos = physics.get_agent_position("human")
            dx, dy = 0, 0
            if keys['w'] or keys['up']: dy -= move_speed
            if keys['s'] or keys['down']: dy += move_speed
            if keys['a'] or keys['left']: dx -= move_speed
            if keys['d'] or keys['right']: dx += move_speed
            
            if dx != 0 and dy != 0:
                dx *= 0.707
                dy *= 0.707
            
            target_x = max(30, min(width - 30, human_pos[0] + dx * 10))
            target_y = max(30, min(height - 30, human_pos[1] + dy * 10))
            physics.set_agent_target("human", target_x, target_y)
            
            # AI update using primitives
            ai_controller.update()
            
            # Physics step
            physics.step()
            
            # Scoring
            for goal in goals:
                for name, body in physics.objects.items():
                    was_inside = name in goal.objects_inside
                    is_inside = goal.contains_point(body.position.x, body.position.y)
                    
                    if is_inside:
                        goal.objects_inside.add(name)
                    else:
                        goal.objects_inside.discard(name)
                    
                    if is_inside and not was_inside and name not in scored_objects:
                        scored_objects.add(name)
                        if goal.assigned_to == "human":
                            human_score += 100
                            print(f"[Score] Human scored with {name}! Total: {human_score}")
                        elif goal.assigned_to == "ai":
                            ai_score += 100
                            print(f"[Score] AI scored with {name}! Total: {ai_score}")
            
            # Render
            renderer.render(
                physics.space,
                goals=goals,
                human_score=human_score,
                ai_score=ai_score,
                tick=tick,
            )
            
            # Labels and messages
            human_pos = physics.get_agent_position("human")
            ai_pos = physics.get_agent_position("ai")
            
            renderer.draw_label(human_pos[0], human_pos[1], "Human", offset_y=25)
            renderer.draw_label(ai_pos[0], ai_pos[1], "AI", offset_y=25)
            
            if ai_primitives.message:
                renderer.draw_speech_bubble(ai_pos[0], ai_pos[1], ai_primitives.message, radius=18)
            
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
