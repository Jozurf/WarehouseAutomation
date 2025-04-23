from q_learning import QLearningAgent
import os

# Simplified Robot class to be used for Q-learning
class Robot:
    def __init__(self, x, y, grid, color=(255, 255, 0)):
        self.x = x
        self.y = y
        self.grid = grid  # Reference to the environment grid
        self.isHoldingPackage = False
        self.color = self.getColor()
        self.path = [(x, y)]  # Track the path for evaluation
        self.waypoint = []
        self.capacity = 1  # Default capacity, will be set randomly in warehouse.py
        self.current_load = 0  # Current load being carried
        self.q_agent = QLearningAgent()
        self.action_dir_map = {
            (0, -1): "UP",
            (0, 1): "DOWN",
            (-1, 0): "LEFT",
            (1, 0): "RIGHT",
        }
        self.dir_alignment = {
            "UP":     {"UP": 0.1, "LEFT": 0.05, "RIGHT": 0.05, "DOWN": -0.05},
            "DOWN":   {"DOWN": 0.1, "LEFT": 0.05, "RIGHT": 0.05, "UP": -0.05},
            "LEFT":   {"LEFT": 0.1, "UP": 0.05, "DOWN": 0.05, "RIGHT": -0.05},
            "RIGHT":  {"RIGHT": 0.1, "UP": 0.05, "DOWN": 0.05, "LEFT": -0.05},
        }
        
        # Try to load weights from q_learning_weights.json
        if os.path.exists('q_learning_weights.json'):
            self.q_agent.load('q_learning_weights.json')
            
        self.last_state = None
        self.last_action = None

    def get_valid_moves(self):
        """Returns a list of valid moves based on the environment grid."""
        possible_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        valid_moves = []
        for dx, dy in possible_moves:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < len(self.grid[0]) and 0 <= new_y < len(self.grid):
                if self.grid[new_y][new_x] != 1:  # Cannot move into a shelf
                    valid_moves.append((dx, dy))
        return valid_moves

    def move(self):
        """Use Q-learning to make movement decisions."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return
            
        current_state = self.q_agent.get_state(self, self.grid)
        action = self.q_agent.get_action(current_state, valid_moves)
        
        if action:
            dx, dy = action
            new_x, new_y = self.x + dx, self.y + dy
            
            # Calculate reward based on collision avoidance
            reward = 0

            # Negative reward for getting too close to obstacles
            adjacent_obstacles = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                                  if 0 <= new_x + dx < len(self.grid[0]) and 
                                     0 <= new_y + dy < len(self.grid) and
                                     self.grid[new_y + dy][new_x + dx] in [1, 2])  # Check for both walls (1) and other robots (2)
            reward -= adjacent_obstacles * 0.5
            
            # Positive reward for successful movement
            reward += 0.1

            # Positive reward for heading towards waypoint if there is one
            curr_state_list = list(current_state)
            dir_to_waypoint = curr_state_list[-1]
            if dir_to_waypoint:
                action_dir = self.action_dir_map.get((dx, dy), None)
                if action_dir in self.dir_alignment.get(dir_to_waypoint, {}):
                    reward += self.dir_alignment[dir_to_waypoint][action_dir]

            # Update position
            self.x = new_x
            self.y = new_y

            # if position is at waypoint, just remove to indicate ready for next waypoint
            if new_x == self.waypoint[0][0] and new_y == self.waypoint[0][1]:
                self.waypoint = None
            self.path.append((self.x, self.y))
            
            # Update Q-values if we're training
            if self.last_state is not None and self.last_action is not None:
                self.q_agent.update(self.last_state, self.last_action, reward, current_state)
            
            self.last_state = current_state
            self.last_action = action
    
    def getColor(self):
        """Returns the color of the robot."""
        if self.isHoldingPackage:
            return (0, 200, 0)
        return (200, 200, 0)
        
    def save_q_learning_weights(self):
        """Save the Q-learning weights to file."""
        self.q_agent.save('q_learning_weights.json')
