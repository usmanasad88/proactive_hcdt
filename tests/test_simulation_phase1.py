#!/usr/bin/env python3
"""
Phase 1 Test: Enhanced 2D environment with multiple shapes and goals.

This script creates a simulation with:
- Human agent (blue) controlled by WASD/Arrow keys
- AI agent (green) that follows a simple helper pattern
- Multiple pushable shapes: Boxes, L-shapes, T-shapes
- Two goal zones: one for human, one for AI
- Compliant push dynamics (objects move smoothly, don't fly away)

Run with:
    cd /home/mani/Repos/proactive_hcdt
    conda activate ur5_python
    python tests/test_simulation_phase1.py
"""

import sys
import math
import random

# Add parent to path for imports
sys.path.insert(0, '/home/mani/Repos/proactive_hcdt')

from proactive_hcdt.simulation import (
    World, WorldConfig, Renderer, GoalZone,
    create_box, create_l_shape, create_t_shape, ShapeType
)


def setup_game_world(world: World) -> None:
    """Set up a game scenario with multiple shapes and two goals."""
    
    # Create goal zones - one for each agent
    human_goal = GoalZone(
        name="Blue Zone",
        x=120,
        y=world.height / 2,
        width=140,
        height=200,
        assigned_to="human",
        target_shapes=[ShapeType.BOX, ShapeType.L_SHAPE, ShapeType.T_SHAPE]
    )
    world.add_goal(human_goal)
    
    ai_goal = GoalZone(
        name="Green Zone",
        x=world.width - 120,
        y=world.height / 2,
        width=140,
        height=200,
        assigned_to="ai",
        target_shapes=[ShapeType.BOX, ShapeType.L_SHAPE, ShapeType.T_SHAPE]
    )
    world.add_goal(ai_goal)
    
    # Create objects in the center area
    center_x = world.width / 2
    center_y = world.height / 2
    
    # Boxes (orange) - 2 pieces
    world.add_object(create_box(
        "Box-1", 
        center_x - 80, center_y - 100,
        size=45,
        color=(255, 165, 0)  # Orange
    ))
    world.add_object(create_box(
        "Box-2",
        center_x + 80, center_y + 100,
        size=45,
        color=(255, 140, 0)  # Dark orange
    ))
    
    # L-shapes (blue) - 2 pieces
    world.add_object(create_l_shape(
        "L-1",
        center_x - 60, center_y + 80,
        size=50,
        color=(100, 149, 237),  # Cornflower blue
        rotation=random.uniform(-0.5, 0.5)
    ))
    world.add_object(create_l_shape(
        "L-2",
        center_x + 100, center_y - 60,
        size=50,
        color=(70, 130, 180),  # Steel blue
        rotation=random.uniform(-0.5, 0.5)
    ))
    
    # T-shapes (red/crimson) - 2 pieces
    world.add_object(create_t_shape(
        "T-1",
        center_x, center_y - 50,
        size=55,
        color=(220, 20, 60),  # Crimson
        rotation=random.uniform(-0.3, 0.3)
    ))
    world.add_object(create_t_shape(
        "T-2",
        center_x + 40, center_y + 30,
        size=55,
        color=(178, 34, 34),  # Firebrick
        rotation=random.uniform(-0.3, 0.3)
    ))


class CooperativeAIBehavior:
    """
    AI behavior that tries to help by pushing objects to its goal.
    
    Strategy:
    1. Observe world state
    2. Find objects not yet in a goal
    3. Pick the closest unattended object
    4. Push it towards the AI's goal zone
    5. Communicate intentions to human
    """
    
    def __init__(self, world: World):
        self.world = world
        self.target_object: str = None
        self.state = "scanning"
        self.state_ticks = 0
        self.last_message_tick = 0
    
    def update(self) -> None:
        """Update AI behavior."""
        ai = self.world.ai_agent
        human = self.world.human_agent
        tick = self.world.tick
        self.state_ticks += 1
        
        # Find the AI's goal zone
        ai_goal = None
        for goal in self.world.goals:
            if goal.assigned_to == "ai":
                ai_goal = goal
                break
        
        if not ai_goal:
            ai.stop()
            return
        
        if self.state == "scanning":
            ai.stop()
            
            # Initial greeting
            if tick < 90:
                if tick == 30:
                    ai.say("Hi! I'll help push objects to the green zone!")
                return
            
            # Find an object to target
            best_obj = None
            best_score = float('inf')
            
            for obj in self.world.objects:
                # Skip if already in a goal
                in_goal = False
                for goal in self.world.goals:
                    if obj.name in goal.objects_inside:
                        in_goal = True
                        break
                if in_goal:
                    continue
                
                # Calculate score (prefer objects closer to AI but far from human)
                dist_to_ai = math.sqrt((obj.x - ai.x)**2 + (obj.y - ai.y)**2)
                dist_to_human = math.sqrt((obj.x - human.x)**2 + (obj.y - human.y)**2)
                
                # AI prefers objects that human is NOT close to
                score = dist_to_ai - dist_to_human * 0.3
                
                if score < best_score:
                    best_score = score
                    best_obj = obj
            
            if best_obj:
                self.target_object = best_obj.name
                self.state = "approaching"
                self.state_ticks = 0
                if tick - self.last_message_tick > 180:
                    ai.say(f"I'll get {best_obj.name}!")
                    self.last_message_tick = tick
            else:
                # All objects in goals
                if tick - self.last_message_tick > 300:
                    ai.say("Great teamwork! 🎉")
                    self.last_message_tick = tick
        
        elif self.state == "approaching":
            obj = self.world.get_object_by_name(self.target_object)
            if not obj:
                self.state = "scanning"
                return
            
            # Check if object reached goal
            if obj.name in ai_goal.objects_inside:
                self.state = "scanning"
                self.target_object = None
                if tick - self.last_message_tick > 120:
                    ai.say("Got one! ✓")
                    self.last_message_tick = tick
                return
            
            # Calculate push position (opposite side of goal from object)
            dx = ai_goal.x - obj.x
            dy = ai_goal.y - obj.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Position behind the object relative to goal
                push_offset = obj.get_bounding_radius() + ai.radius + 5
                target_x = obj.x - (dx / dist) * push_offset
                target_y = obj.y - (dy / dist) * push_offset
            else:
                target_x, target_y = obj.x, obj.y
            
            # Move towards push position
            dist_to_target = ai.distance_to_point(target_x, target_y)
            
            if dist_to_target > 10:
                ai.move_towards(target_x, target_y, arrival_threshold=8)
            else:
                # In position, push towards goal
                self.state = "pushing"
                self.state_ticks = 0
        
        elif self.state == "pushing":
            obj = self.world.get_object_by_name(self.target_object)
            if not obj:
                self.state = "scanning"
                return
            
            # Check if object reached goal
            if obj.name in ai_goal.objects_inside:
                self.state = "scanning"
                self.target_object = None
                ai.stop()
                return
            
            # Push towards goal
            dx = ai_goal.x - obj.x
            dy = ai_goal.y - obj.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 5:
                # Move towards object (will push it)
                ai.move_towards(obj.x, obj.y, arrival_threshold=5)
            else:
                # Object is in goal area
                self.state = "scanning"
            
            # Timeout - try different approach
            if self.state_ticks > 300:
                self.state = "scanning"
                self.target_object = None
        
        # Clear messages after a while
        if ai.message and tick - self.last_message_tick > 120:
            ai.clear_message()


def main():
    """Run the enhanced Phase 1 test simulation."""
    print("=" * 60)
    print("Phase 1 Test: Multi-Shape Collaborative Environment")
    print("=" * 60)
    print()
    print("Controls:")
    print("  WASD or Arrow Keys: Move the blue (Human) agent")
    print("  R: Reset simulation")
    print("  ESC: Quit")
    print()
    print("Objective:")
    print("  - Push shapes to your BLUE goal zone (left side)")
    print("  - The AI will push shapes to its GREEN goal zone (right)")
    print("  - Shapes: Boxes (orange), L-shapes (blue), T-shapes (red)")
    print()
    print("Physics: Compliant pushing - objects move smoothly with you!")
    print()
    
    # Create world with larger size for more objects
    config = WorldConfig(
        width=1000,
        height=700,
        title="Human-AI Collaboration: Push Shapes to Goals!"
    )
    world = World(config)
    
    # Setup game
    setup_game_world(world)
    
    # Create renderer
    renderer = Renderer(world)
    renderer.initialize()
    
    # Create AI behavior
    ai_behavior = CooperativeAIBehavior(world)
    
    print("Starting simulation...")
    print()
    
    try:
        while world.running:
            # Process events and get keyboard state
            keys = renderer.process_events()
            
            # Handle reset
            if keys.get('r'):
                print("Resetting simulation...")
                world.reset()
                setup_game_world(world)
                ai_behavior = CooperativeAIBehavior(world)
            
            # Update human agent based on keyboard
            world.human_agent.handle_keyboard(keys)
            
            # Update AI behavior
            ai_behavior.update()
            
            # Render and get delta time
            dt = renderer.render()
            
            # Update world physics
            world.update(dt)
            
            # Check for game completion
            total_objects = len(world.objects)
            objects_in_goals = sum(len(g.objects_inside) for g in world.goals)
            
            if objects_in_goals == total_objects and total_objects > 0:
                if world.tick % 60 == 0:  # Don't spam
                    print(f"All {total_objects} objects in goals!")
                    print(f"Human score: {world.human_score} | AI score: {world.ai_score}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        renderer.cleanup()
        print()
        print("=" * 40)
        print("Final Scores:")
        print(f"  Human: {world.human_score}")
        print(f"  AI:    {world.ai_score}")
        print(f"  Total: {world.score}")
        print("=" * 40)
        print("Simulation ended.")


if __name__ == "__main__":
    main()
