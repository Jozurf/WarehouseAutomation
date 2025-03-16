import pygame
import sys
from robot import Robot  # Import the Robot class

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse Q-learning Environment")

# Define colors
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (200, 200, 200)
BLUE   = (100, 100, 255)
GREEN  = (0, 255, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)

# Grid settings
ROWS, COLS = 20, 20
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)

# Define environment grid
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Create boundary walls
for i in range(ROWS):
    grid[i][0] = 1
    grid[i][COLS - 1] = 1
for j in range(COLS):
    grid[0][j] = 1
    grid[ROWS - 1][j] = 1

# Add inner walls with varying spaces
for i in range(3, 17):
    grid[i][5] = 1
    grid[i][14] = 1

# Define lanes
for j in range(1, COLS - 1):
    grid[2][j] = 3
    grid[10][j] = 2

# Set pickup and dropoff zones
pickup_zone = [(1, 1)]
for i, j in pickup_zone:
    grid[i][j] = 4
dropoff_zone = [(ROWS - 2, COLS - 2)]
for i, j in dropoff_zone:
    grid[i][j] = 5

# Instantiate a robot at the pickup area
robot = Robot(1, 1, grid)

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(5)  # Control the frame rate
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update robot movement using the logic from robot.py
    robot.move()

    # Check if the robot is holding a package
    if (robot.x, robot.y) in pickup_zone:
        robot.isHoldingPackage = True
    elif (robot.x, robot.y) in dropoff_zone:
        robot.isHoldingPackage = False

    # Draw the grid and environment
    win.fill(WHITE)
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_value = grid[i][j]
            if cell_value == 1:
                pygame.draw.rect(win, BLACK, rect)
            elif cell_value == 2:
                pygame.draw.rect(win, GRAY, rect)
            elif cell_value == 3:
                pygame.draw.rect(win, BLUE, rect)
            elif cell_value == 4:
                pygame.draw.rect(win, GREEN, rect)
            elif cell_value == 5:
                pygame.draw.rect(win, RED, rect)
            else:
                pygame.draw.rect(win, WHITE, rect)
            pygame.draw.rect(win, BLACK, rect, 1)

    # Draw the robot
    cx = robot.x * CELL_SIZE + CELL_SIZE // 2
    cy = robot.y * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(win, robot.color, (cx, cy), CELL_SIZE // 3)

    pygame.display.update()

pygame.quit()
sys.exit()