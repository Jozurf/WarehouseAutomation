import pygame
import sys
import heapq
from robot import Robot

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
PURPLE = (128, 0, 128)

# Grid settings
ROWS, COLS = 20, 20
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)

# Define environment grid
# 0: Empty, 1: Wall, 4: Pickup, 5: Dropoff
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Create boundary walls
for i in range(ROWS):
    grid[i][0] = 1
    grid[i][COLS - 1] = 1
for j in range(COLS):
    grid[0][j] = 1
    grid[ROWS - 1][j] = 1

# Add inner walls
for i in range(3, 17):
    grid[i][5] = 1
    grid[i][14] = 1

grid[10][5] = 0  # Create a small passage in the left wall
grid[10][14] = 0  # Create a small passage in the right wall

# Define pickup and dropoff zones
PICKUP = (3, 3)
DROPOFF = (16, 16)
grid[PICKUP[0]][PICKUP[1]] = 4
grid[DROPOFF[0]][DROPOFF[1]] = 5

# A* Pathfinding Algorithm
def astar_search(start, goal, grid):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)  # Add the start position
            return path[::-1]  # Return reversed path
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Down, Right, Up, Left
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and grid[neighbor[0]][neighbor[1]] != 1:
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    if all(item[1] != neighbor for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found

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

def main():
    # Initialize robot at starting position
    robot_pos = (1, 1)  # Start position
    robot = Robot(robot_pos[1], robot_pos[0], grid)
    
    # Create paths
    to_pickup_path = astar_search(robot_pos, PICKUP, grid)
    to_dropoff_path = []
    
    # Game state
    clock = pygame.time.Clock()
    path_index = 0
    current_path = to_pickup_path
    has_pickup = False
    
    # Font for status messages
    font = pygame.font.SysFont(None, 36)
    
    running = True
    while running:
        clock.tick(5)  # Control the speed of the simulation
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Move robot along the current path
        if path_index < len(current_path):
            next_pos = current_path[path_index]
            robot.x = next_pos[1]
            robot.y = next_pos[0]
            robot.path.append((robot.x, robot.y))
            path_index += 1
            
            # Check if robot is at pickup location
            if not has_pickup and (robot.y, robot.x) == PICKUP:
                robot.isHoldingPackage = True
                robot.color = robot.getColor()  # Update color based on package status
                has_pickup = True
                print("Robot picked up the package!")
                
                # Calculate path to dropoff
                current_path = astar_search(PICKUP, DROPOFF, grid)
                path_index = 0
                
            # Check if robot is at dropoff location with package
            elif robot.isHoldingPackage and (robot.y, robot.x) == DROPOFF:
                robot.isHoldingPackage = False
                robot.color = robot.getColor()  # Update color based on package status
                print("Package delivered successfully!")
        
        # Draw everything
        win.fill(WHITE)
        draw_grid()
        
        # Draw the current path
        draw_path(current_path)
        
        # Draw robot
        robot_rect = pygame.Rect(robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(win, robot.color, robot_rect)
        
        # Draw robot's path
        for x, y in robot.path:
            if (x, y) != (robot.x, robot.y):  # Don't draw over the robot's current position
                path_rect = pygame.Rect(x * CELL_SIZE + CELL_SIZE//3, y * CELL_SIZE + CELL_SIZE//3, CELL_SIZE//3, CELL_SIZE//3)
                pygame.draw.rect(win, PURPLE, path_rect)
        
        # Display status message
        status_text = "Status: "
        if robot.isHoldingPackage:
            status_text += "Carrying package to dropoff"
        elif has_pickup:
            status_text += "Package delivered"
        else:
            status_text += "Moving to pickup"
            
        status_surface = font.render(status_text, True, BLACK)
        win.blit(status_surface, (10, HEIGHT - 40))
        
        pygame.display.update()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()