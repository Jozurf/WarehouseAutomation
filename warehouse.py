import pygame
import sys
import random
import argparse
from robot import Robot
from OneRobotAStarAgent import OneRobotAStarAgent

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
ORANGE = (255, 165, 0)  # Robot color when carrying package
PURPLE = (128, 0, 128)

# Grid settings
ROWS, COLS = 20, 20
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)

# Global variables
grid = None
pickup_locations = []
dropoff_locations = []

def generate_random_grid(num_pickups=1, num_dropoffs=1):
    # Initialize empty grid
    new_grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    
    # Create boundary walls
    for i in range(ROWS):
        new_grid[i][0] = 1
        new_grid[i][COLS - 1] = 1
    for j in range(COLS):
        new_grid[0][j] = 1
        new_grid[ROWS - 1][j] = 1
    
    # Add random internal walls (avoiding corners)
    wall_chance = 0.15  # Probability of a cell being a wall
    for i in range(2, ROWS-2):
        for j in range(2, COLS-2):
            if random.random() < wall_chance:
                new_grid[i][j] = 1
    
    # Create paths through the grid
    # Horizontal corridors at 1/4, 1/2, and 3/4 of grid height
    for j in range(1, COLS-1):
        new_grid[ROWS//4][j] = 0
        new_grid[ROWS//2][j] = 0
        new_grid[3*ROWS//4][j] = 0
    
    # Vertical corridors at 1/4, 1/2, and 3/4 of grid width
    for i in range(1, ROWS-1):
        new_grid[i][COLS//4] = 0
        new_grid[i][COLS//2] = 0
        new_grid[i][3*COLS//4] = 0
    
    # Generate pickup and dropoff locations
    pickups = []
    dropoffs = []
    
    for _ in range(num_pickups):
        while True:
            row = random.randint(2, ROWS-3)
            col = random.randint(2, COLS-3)
            
            # Ensure it's not on a wall and not already a pickup or dropoff
            if new_grid[row][col] == 0:
                new_grid[row][col] = 4  # Mark as pickup
                pickups.append((row, col))
                break
    
    for _ in range(num_dropoffs):
        while True:
            row = random.randint(2, ROWS-3)
            col = random.randint(2, COLS-3)
            
            # Ensure it's not on a wall and not already a pickup or dropoff
            if new_grid[row][col] == 0:
                new_grid[row][col] = 5  # Mark as dropoff
                dropoffs.append((row, col))
                break
    
    return new_grid, pickups, dropoffs

def find_valid_start_position(search_grid):
    """Find a valid starting position that's not a wall, pickup, or dropoff."""
    while True:
        row = random.randint(1, ROWS-2)
        col = random.randint(1, COLS-2)
        if search_grid[row][col] == 0:  # Empty space
            return (row, col)

def draw_grid():
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[i][j] == 1:  # Wall
                pygame.draw.rect(win, BLACK, rect)
            elif grid[i][j] == 4:  # Pickup
                pygame.draw.rect(win, GREEN, rect)
            elif grid[i][j] == 5:  # Dropoff
                pygame.draw.rect(win, RED, rect)
            else:  # Empty
                pygame.draw.rect(win, WHITE, rect)
            pygame.draw.rect(win, GRAY, rect, 1)  # Grid lines

def draw_path(path):
    for i, (row, col) in enumerate(path):
        # Skip the start and end points
        if i > 0 and i < len(path) - 1:
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(win, BLUE, rect, 2)

def run_simulation(num_pickups, num_dropoffs):
    global grid, pickup_locations, dropoff_locations
    
    # Find a valid starting position
    robot_pos = find_valid_start_position(grid)
    robot = Robot(robot_pos[1], robot_pos[0], grid)
    
    # Create A* agent
    agent = OneRobotAStarAgent(grid, robot_pos, pickup_locations[0], dropoff_locations[0])
    
    # Plan the path
    if not agent.plan_path():
        print("Could not plan a valid path! Regenerating grid...")
        return True  # Restart the simulation
    
    # Override robot's getColor method to use our colors
    def custom_get_color(self):
        if self.isHoldingPackage:
            return ORANGE  # Different color when carrying package
        return YELLOW
    
    Robot.getColor = custom_get_color
    
    # Game state
    clock = pygame.time.Clock()
    
    # Font for status messages
    font = pygame.font.SysFont(None, 36)
    
    running = True
    while running:
        clock.tick(5)  # Control the speed of the simulation
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press R to regenerate the grid
                    return True  # Restart the simulation
        
        # Move robot along the planned path
        if not agent.has_completed_path():
            next_pos = agent.get_next_move()
            if next_pos:
                robot.y = next_pos[0]
                robot.x = next_pos[1]
                robot.path.append((robot.x, robot.y))
                
                # Update robot's package status
                robot.isHoldingPackage = agent.is_holding_package
        
        # Draw everything
        win.fill(WHITE)
        draw_grid()
        
        # Draw the current path
        draw_path(agent.current_path)
        
        # Draw robot
        robot_rect = pygame.Rect(robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(win, robot.getColor(), robot_rect)
        
        # Draw robot's path
        for x, y in robot.path:
            if (x, y) != (robot.x, robot.y):  # Don't draw over the robot's current position
                path_rect = pygame.Rect(x * CELL_SIZE + CELL_SIZE//3, y * CELL_SIZE + CELL_SIZE//3, CELL_SIZE//3, CELL_SIZE//3)
                pygame.draw.rect(win, PURPLE, path_rect)
        
        # Display status message
        status_text = "Status: "
        if robot.isHoldingPackage:
            status_text += "Carrying package to dropoff"
        elif agent.has_completed_path():
            status_text += "Package delivered"
        else:
            status_text += "Moving to pickup"
            
        status_surface = font.render(status_text, True, BLACK)
        win.blit(status_surface, (10, HEIGHT - 40))
        
        # Display instructions
        instructions = "Press R to regenerate grid"
        instr_surface = font.render(instructions, True, BLACK)
        win.blit(instr_surface, (WIDTH - 300, HEIGHT - 40))
        
        pygame.display.update()
        
        # Check if simulation is complete
        if agent.has_completed_path():
            pygame.time.delay(2000)  # Wait 2 seconds before allowing restart
    
    return False

def main():
    global grid, pickup_locations, dropoff_locations
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Warehouse A* Navigation Simulation')
    parser.add_argument('--agent', type=str, default='onerobotastar', help='Agent type: simple or onerobotastar')
    parser.add_argument('--pickups', type=int, default=1, help='Number of pickup locations')
    parser.add_argument('--dropoffs', type=int, default=1, help='Number of dropoff locations')
    args = parser.parse_args()
    
    # Check if we're using the OneRobotA* agent
    if args.agent.lower() != 'onerobotastar':
        print("Please use --agent onerobotastar to use the A* agent")
        return
    
    # Initialize grid and pickup/dropoff points
    grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)
    
    # Run the simulation loop
    while run_simulation(args.pickups, args.dropoffs):
        grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)

if __name__ == '__main__':
    main()