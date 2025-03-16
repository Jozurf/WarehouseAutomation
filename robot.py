import random

class Robot:
    def __init__(self, x, y, grid, color=(255, 255, 0)):
        self.x = x
        self.y = y
        self.grid = grid  # Reference to the environment grid
        self.color = color
        self.path = [(x, y)]  # Track the path for evaluation

    def get_valid_moves(self):
        """Returns a list of valid moves based on the environment grid."""
        possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Down, Right, Up, Left
        valid_moves = []
        for dx, dy in possible_moves:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < len(self.grid[0]) and 0 <= new_y < len(self.grid):
                if self.grid[new_y][new_x] != 1:  # Cannot move into a shelf
                    valid_moves.append((dx, dy))
        return valid_moves

    def move(self):
        """Replace this logic with Q-learning."""
        valid_moves = self.get_valid_moves()
        if valid_moves:
            dx, dy = random.choice(valid_moves)  # Temporary: Replace with Q-learning decision
            self.x += dx
            self.y += dy
            self.path.append((self.x, self.y))
