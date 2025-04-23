import pygame
import sys
import random
import argparse
from robot import Robot
from MultiRobotAgent import MultiRobotAgent

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse A* Navigation")

start_pos = None

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
CYAN = (0, 255, 255)

# Robot colors - unique for each robot (no green)
ROBOT_COLORS = [
    (255, 100, 100),  # Light red
    (100, 100, 255),  # Light blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Pink
    (100, 70, 70),    # Dark red
    (200, 150, 100),  # Brown
    (150, 100, 200),  # Purple
    (200, 100, 150),  # Rose
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Deep Purple
]

# Grid settings
ROWS, COLS = 20, 20
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)

# Global variables
grid = None
pickup_locations = []
dropoff_locations = []
completed_pickups = set()  # Track completed pickups
package_sizes = {}  # Store package sizes

def generate_random_grid(num_pickups=1, num_dropoffs=1):
    new_grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for i in range(ROWS):
        new_grid[i][0] = 1
        new_grid[i][COLS - 1] = 1
    for j in range(COLS):
        new_grid[0][j] = 1
        new_grid[ROWS - 1][j] = 1

    wall_chance = 0.15
    for i in range(2, ROWS - 2):
        for j in range(2, COLS - 2):
            if random.random() < wall_chance:
                new_grid[i][j] = 1

    for j in range(1, COLS - 1):
        new_grid[ROWS // 4][j] = 0
        new_grid[ROWS // 2][j] = 0
        new_grid[3 * ROWS // 4][j] = 0
    for i in range(1, ROWS - 1):
        new_grid[i][COLS // 4] = 0
        new_grid[i][COLS // 2] = 0
        new_grid[i][3 * COLS // 4] = 0

    pickups = []
    dropoffs = []
    for _ in range(num_pickups):
        while True:
            row = random.randint(2, ROWS - 3)
            col = random.randint(2, COLS - 3)
            if new_grid[row][col] == 0:
                new_grid[row][col] = 4
                pickups.append((row, col))
                package_sizes[(row, col)] = random.randint(1, 3)  # Random size 1-3
                break
    for _ in range(num_dropoffs):
        while True:
            row = random.randint(2, ROWS - 3)
            col = random.randint(2, COLS - 3)
            if new_grid[row][col] == 0:
                new_grid[row][col] = 5
                dropoffs.append((row, col))
                break
    return new_grid, pickups, dropoffs, package_sizes

def find_valid_start_position(search_grid):
    while True:
        row = random.randint(1, ROWS - 2)
        col = random.randint(1, COLS - 2)
        if search_grid[row][col] == 0:
            return (row, col)

def draw_grid():
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell = grid[i][j]

            if (i, j) == start_pos:
                pygame.draw.rect(win, CYAN, rect)  # Charging station
            elif cell == 1:
                pygame.draw.rect(win, BLACK, rect)
            elif cell == 4:  # Pickup location
                if (i, j) in completed_pickups:
                    pygame.draw.rect(win, WHITE, rect)
                else:
                    pygame.draw.rect(win, GREEN, rect)
                    # Draw package size
                    if (i, j) in package_sizes:
                        size_text = str(package_sizes[(i, j)])
                        text_surface = font.render(size_text, True, BLACK)
                        text_rect = text_surface.get_rect(center=(j * CELL_SIZE + CELL_SIZE//2, 
                                                            i * CELL_SIZE + CELL_SIZE//2))
                        win.blit(text_surface, text_rect)
            elif cell == 5:
                pygame.draw.rect(win, RED, rect)
            else:
                pygame.draw.rect(win, WHITE, rect)
            pygame.draw.rect(win, GRAY, rect, 1)

def run_simulation(num_pickups, num_dropoffs, num_robots):
    global grid, pickup_locations, dropoff_locations, start_pos, completed_pickups, package_sizes, font
    robot_pos = find_valid_start_position(grid)
    start_pos = robot_pos
    completed_pickups = set()  # Track completed pickups

    agent = MultiRobotAgent(grid, robot_pos, pickup_locations, dropoff_locations, num_robots, package_sizes)
    agent.assign_initial_pickups()
    print("Assigned initial pickups to all robots.")
    robots = [Robot(robot_pos[1], robot_pos[0], grid) for _ in range(num_robots)]

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Font for package number
    
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

        win.fill(WHITE)
        draw_grid()

        # Draw robots with their capacities
        for i, robot in enumerate(robots):
            robot_rect = pygame.Rect(robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(win, ROBOT_COLORS[i % len(ROBOT_COLORS)], robot_rect)
            
            agent_robot = agent.robots[i]
            capacity_text = f"{agent_robot.current_load}/{agent_robot.capacity}"
            
            text_surface = font.render(capacity_text, True, BLACK)
            text_rect = text_surface.get_rect(center=(robot.x * CELL_SIZE + CELL_SIZE//2, 
                                                    robot.y * CELL_SIZE + CELL_SIZE//2))
            win.blit(text_surface, text_rect)

        moves = agent.get_next_moves()
        for i, move in enumerate(moves):
            if move:
                robots[i].y, robots[i].x = move
                robots[i].path.append((robots[i].x, robots[i].y))
                # Let MultiRobotAgent handle completed_pickups
                completed_pickups.update(agent.completed_pickups)

        status_text = f"Robots: {len(robots)}"
        status_surface = font.render(status_text, True, BLACK)
        win.blit(status_surface, (10, HEIGHT - 30))

        pygame.display.flip()

        if agent.all_tasks_done():
            print("All tasks completed!")
            return False

    return False

def main():
    global grid, pickup_locations, dropoff_locations, package_sizes, font

    parser = argparse.ArgumentParser(description='Warehouse A* Navigation Simulation')
    parser.add_argument('--pickups', type=int, default=1, help='Number of pickup locations')
    parser.add_argument('--dropoffs', type=int, default=1, help='Number of dropoff locations')
    parser.add_argument('--robots', type=int, default=2, help='Number of robots')
    args = parser.parse_args()

    font = pygame.font.SysFont(None, 24)  # Initialize font
    grid, pickup_locations, dropoff_locations, package_sizes = generate_random_grid(args.pickups, args.dropoffs)
    while run_simulation(args.pickups, args.dropoffs, args.robots):
        grid, pickup_locations, dropoff_locations, package_sizes = generate_random_grid(args.pickups, args.dropoffs)

if __name__ == '__main__':
    main()
