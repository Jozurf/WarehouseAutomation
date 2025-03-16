import pygame
import sys
import random
import argparse
from robot import Robot
from OneRobotAStarAgent import OneRobotAStarAgent
from heapq import heappush, heappop

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse A* Navigation")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 100, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Grid settings
ROWS, COLS = 20, 20
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)

# Global variables
grid = None
pickup_locations = []
dropoff_locations = []

def generate_random_grid(num_pickups=1, num_dropoffs=1):
    new_grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for i in range(ROWS):
        new_grid[i][0] = 1
        new_grid[i][COLS - 1] = 1
    for j in range(COLS):
        new_grid[0][j] = 1
        new_grid[ROWS - 1][j] = 1
    
    wall_chance = 0.15
    for i in range(2, ROWS-2):
        for j in range(2, COLS-2):
            if random.random() < wall_chance:
                new_grid[i][j] = 1
    
    for j in range(1, COLS-1):
        new_grid[ROWS//4][j] = 0
        new_grid[ROWS//2][j] = 0
        new_grid[3*ROWS//4][j] = 0
    for i in range(1, ROWS-1):
        new_grid[i][COLS//4] = 0
        new_grid[i][COLS//2] = 0
        new_grid[i][3*COLS//4] = 0
    
    pickups = []
    dropoffs = []
    for _ in range(num_pickups):
        while True:
            row = random.randint(2, ROWS-3)
            col = random.randint(2, COLS-3)
            if new_grid[row][col] == 0:
                new_grid[row][col] = 4
                pickups.append((row, col))
                break
    for _ in range(num_dropoffs):
        while True:
            row = random.randint(2, ROWS-3)
            col = random.randint(2, COLS-3)
            if new_grid[row][col] == 0:
                new_grid[row][col] = 5
                dropoffs.append((row, col))
                break
    return new_grid, pickups, dropoffs

def find_valid_start_position(search_grid):
    while True:
        row = random.randint(1, ROWS-2)
        col = random.randint(1, COLS-2)
        if search_grid[row][col] == 0:
            return (row, col)

def draw_grid():
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[i][j] == 1:
                pygame.draw.rect(win, BLACK, rect)
            elif grid[i][j] == 4:
                pygame.draw.rect(win, GREEN, rect)
            elif grid[i][j] == 5:
                pygame.draw.rect(win, RED, rect)
            else:
                pygame.draw.rect(win, WHITE, rect)
            pygame.draw.rect(win, GRAY, rect, 1)

def draw_path(path):
    for i, (row, col) in enumerate(path):
        if i > 0 and i < len(path) - 1:
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(win, BLUE, rect, 2)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class MultiPickupAgent:
    def __init__(self, grid, start_pos, pickups, dropoffs):
        self.grid = grid
        self.start_pos = start_pos
        self.pickups = pickups.copy()
        self.dropoffs = dropoffs.copy()
        self.current_path = []
        self.is_holding_package = False
        self.completed_pickups = 0
        self.robot_pos = start_pos
        self.path_index = 0
        
    def find_optimal_path(self):
        if not self.pickups and not self.is_holding_package:
            return True
        
        # Priority queue for path planning
        pq = []
        visited = set()
        
        # Start state: (total_cost, cost_so_far, position, pickups_remaining, has_package, path)
        initial_state = (0, 0, self.start_pos, tuple(self.pickups), False, [self.start_pos])
        heappush(pq, initial_state)
        
        while pq:
            total_cost, cost_so_far, pos, pickups_rem, has_package, path = heappop(pq)
            
            state = (pos, pickups_rem, has_package)
            if state in visited:
                continue
            visited.add(state)
            
            # If we've picked up all packages and dropped off
            if not pickups_rem and not has_package:
                self.current_path = path
                return True
            
            # Generate next possible moves
            if has_package:
                # Find nearest dropoff
                for dropoff in self.dropoffs:
                    agent = OneRobotAStarAgent(self.grid, pos, dropoff, dropoff)
                    if agent.plan_path():
                        new_cost = cost_so_far + len(agent.current_path)
                        new_path = path + agent.current_path[1:]
                        heappush(pq, (new_cost, new_cost, dropoff, pickups_rem, False, new_path))
            else:
                # Find nearest pickup
                for pickup in pickups_rem:
                    agent = OneRobotAStarAgent(self.grid, pos, pickup, pickup)
                    if agent.plan_path():
                        new_cost = cost_so_far + len(agent.current_path)
                        new_pickups = tuple(p for p in pickups_rem if p != pickup)
                        new_path = path + agent.current_path[1:]
                        heappush(pq, (new_cost, new_cost, pickup, new_pickups, True, new_path))
        
        return False

    def get_next_move(self):
        if self.path_index < len(self.current_path) - 1:
            self.path_index += 1
            self.robot_pos = self.current_path[self.path_index]
            
            # Check if we're at a pickup or dropoff
            if self.robot_pos in self.pickups and not self.is_holding_package:
                self.is_holding_package = True
                self.pickups.remove(self.robot_pos)
                self.completed_pickups += 1
            elif self.robot_pos in self.dropoffs and self.is_holding_package:
                self.is_holding_package = False
                
            return self.robot_pos
        return None

    def has_completed_path(self):
        return self.path_index >= len(self.current_path) - 1 and not self.pickups and not self.is_holding_package

def run_simulation(num_pickups, num_dropoffs):
    global grid, pickup_locations, dropoff_locations
    
    robot_pos = find_valid_start_position(grid)
    robot = Robot(robot_pos[1], robot_pos[0], grid)
    agent = MultiPickupAgent(grid, robot_pos, pickup_locations, dropoff_locations)
    
    if not agent.find_optimal_path():
        print("Could not plan a valid path! Regenerating grid...")
        return True
    
    def custom_get_color(self):
        if self.isHoldingPackage:
            return ORANGE
        return YELLOW
    
    Robot.getColor = custom_get_color
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    
    running = True
    while running:
        clock.tick(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
        
        if not agent.has_completed_path():
            next_pos = agent.get_next_move()
            if next_pos:
                robot.y = next_pos[0]
                robot.x = next_pos[1]
                robot.path.append((robot.x, robot.y))
                robot.isHoldingPackage = agent.is_holding_package
        
        win.fill(WHITE)
        draw_grid()
        draw_path(agent.current_path)
        
        robot_rect = pygame.Rect(robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(win, robot.getColor(), robot_rect)
        
        for x, y in robot.path:
            if (x, y) != (robot.x, robot.y):
                path_rect = pygame.Rect(x * CELL_SIZE + CELL_SIZE//3, y * CELL_SIZE + CELL_SIZE//3, CELL_SIZE//3, CELL_SIZE//3)
                pygame.draw.rect(win, PURPLE, path_rect)
        
        status_text = f"Status: Pickups completed: {agent.completed_pickups}/{num_pickups}"
        if robot.isHoldingPackage:
            status_text += " - Carrying package"
        elif agent.has_completed_path():
            status_text += " - All packages delivered"
        else:
            status_text += " - Moving to pickup"
            
        status_surface = font.render(status_text, True, BLACK)
        win.blit(status_surface, (10, HEIGHT - 40))
        
        instructions = "Press R to regenerate grid"
        instr_surface = font.render(instructions, True, BLACK)
        win.blit(instr_surface, (WIDTH - 300, HEIGHT - 40))
        
        pygame.display.update()
        
        if agent.has_completed_path():
            pygame.time.delay(2000)
    
    return False

def main():
    global grid, pickup_locations, dropoff_locations
    
    parser = argparse.ArgumentParser(description='Warehouse A* Navigation Simulation')
    parser.add_argument('--pickups', type=int, default=1, help='Number of pickup locations')
    parser.add_argument('--dropoffs', type=int, default=1, help='Number of dropoff locations')
    args = parser.parse_args()
    
    grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)
    while run_simulation(args.pickups, args.dropoffs):
        grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)

if __name__ == '__main__':
    main()