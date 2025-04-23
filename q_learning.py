import numpy as np
import json
import os
import random
import ast

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_state(self, robot, grid):
        """Convert the robot's surroundings into a state representation"""
        # Handle both Robot and RobotAgents classes
        if hasattr(robot, 'position'):  # RobotAgents class
            x, y = robot.position
        else:  # Robot class
            x, y = robot.x, robot.y
            
        # Check 4 adjacent cells for obstacles (1 for obstacle, 0 for free)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        state = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
                state.append(1 if grid[new_y][new_x] in [1, 2] else 0)
            else:
                state.append(1)  # Treat out of bounds as obstacle
        
        # add direction to next waypoint
        if hasattr(robot, 'waypoint'): # robot class
            next_waypoint = robot.waypoint[0][-1] if robot.waypoint and robot.waypoint[0] else None
            if next_waypoint:
                dir_to_waypoint = self.get_direction(x, y, next_waypoint[0], next_waypoint[1])
                state.append(dir_to_waypoint)
            else:
                state.append(None)
        else: # robot agent class
            next_waypoint = robot.path[0][-1] if robot.path and robot.path[0] else None

            if next_waypoint:
                dir_to_waypoint = self.get_direction(x, y, next_waypoint[0], next_waypoint[1])
                state.append(dir_to_waypoint)
            else:
                state.append(None)
        return tuple(state)
    
    def get_action(self, state, valid_moves):
        """Choose an action using epsilon-greedy policy"""
        if not valid_moves:
            return None
        if state not in self.q_table:
            self.q_table[state] = {tuple(move): 0.0 for move in [(0, 1), (1, 0), (0, -1), (-1, 0)]}
            
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        # Get Q-values for valid moves only
        valid_q_values = [(move, self.q_table[state][tuple(move)]) for move in valid_moves]
        return max(valid_q_values, key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """Update Q-value for state-action pair"""
        action = tuple(action)  # Convert action to tuple for dictionary key
        if state not in self.q_table:
            self.q_table[state] = {tuple(move): 0.0 for move in [(0, 1), (1, 0), (0, -1), (-1, 0)]}
        if next_state not in self.q_table:
            self.q_table[next_state] = {tuple(move): 0.0 for move in [(0, 1), (1, 0), (0, -1), (-1, 0)]}
            
        # Get best next action value
        best_next_value = max(self.q_table[next_state].values())
        
        # Update Q-value
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * best_next_value - self.q_table[state][action]
        )
    
    def save(self, filename='q_learning_weights.json'):
        """Save Q-table to file"""
        # Convert dictionary keys to strings for JSON serialization
        serializable_q_table = {}
        for state, actions in self.q_table.items():
            state_str = str(state)
            serializable_q_table[state_str] = {
                str(action): value 
                for action, value in actions.items()
            }
        print(f"Saving Q-table to {filename} with size {len(serializable_q_table)}")
        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f)
    
    def load(self, filename='q_learning_weights.json'):
        """Load Q-table from file"""
        if not os.path.exists(filename):
            return False
            
        with open(filename, 'r') as f:
            serialized_q_table = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = {}
        for state_str, actions in serialized_q_table.items():
            state = ast.literal_eval(state_str)
            self.q_table[state] = {}
            for action_str, value in actions.items():
                action = ast.literal_eval(action_str)
                self.q_table[state][action] = value
        return True
    
    def get_direction(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) > abs(dy):
            return 'RIGHT' if dx > 0 else 'LEFT'
        elif dy != 0:
            return 'DOWN' if dy > 0 else 'UP'
        else:
            return None  # same point