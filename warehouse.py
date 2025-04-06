import pygame
import sys
import random
import argparse
from robot import Robot
from OneRobotAStarAgent import OneRobotAStarAgent
from MultiRobotAgent import MultiRobotAgent
from heapq import heappush, heappop

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
                break
    for _ in range(num_dropoffs):
        while True:
            row = random.randint(2, ROWS - 3)
            col = random.randint(2, COLS - 3)
            if new_grid[row][col] == 0:
                new_grid[row][col] = 5
                dropoffs.append((row, col))
                break
    return new_grid, pickups, dropoffs

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
            elif cell == 4:
                pygame.draw.rect(win, GREEN, rect)
            elif cell == 5:
                pygame.draw.rect(win, RED, rect)
            else:
                pygame.draw.rect(win, WHITE, rect)
            pygame.draw.rect(win, GRAY, rect, 1)

def draw_path(path):
    for i, (row, col) in enumerate(path):
        if i > 0 and i < len(path) - 1:
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(win, BLUE, rect, 2)

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

        pq = []
        visited = set()

        initial_state = (0, 0, self.start_pos, tuple(self.pickups), False, [self.start_pos])
        heappush(pq, initial_state)

        while pq:
            total_cost, cost_so_far, pos, pickups_rem, has_package, path = heappop(pq)
            state = (pos, pickups_rem, has_package)
            if state in visited:
                continue
            visited.add(state)

            if not pickups_rem and not has_package:
                self.current_path = path
                return True

            if has_package:
                for dropoff in self.dropoffs:
                    agent = OneRobotAStarAgent(self.grid, pos, dropoff, dropoff)
                    if agent.plan_path():
                        new_cost = cost_so_far + len(agent.current_path)
                        new_path = path + agent.current_path[1:]
                        heappush(pq, (new_cost, new_cost, dropoff, pickups_rem, False, new_path))
            else:
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

def run_simulation(agent_type, num_pickups, num_dropoffs, num_robots):
    global grid, pickup_locations, dropoff_locations, start_pos
    robot_pos = find_valid_start_position(grid)
    start_pos = robot_pos

    if agent_type == 'multi':
        agent = MultiRobotAgent(grid, robot_pos, pickup_locations, dropoff_locations, num_robots)
        agent.plan_paths()
        print("Planned paths for all robots.")
        robots = [Robot(robot_pos[1], robot_pos[0], grid) for _ in range(num_robots)]
    else:
        agent = MultiPickupAgent(grid, robot_pos, pickup_locations, dropoff_locations)
        if not agent.find_optimal_path():
            print("Could not plan a valid path! Regenerating grid...")
            return True
        robots = [Robot(robot_pos[1], robot_pos[0], grid)]

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

        # Create translucent surface for path blocks
        translucent_surface = pygame.Surface((CELL_SIZE // 3, CELL_SIZE // 3), pygame.SRCALPHA)
        translucent_surface.fill((128, 0, 128, int(0.2 * 255)))  # RGBA

        # Track which path cells were already drawn with transparency
        drawn_translucents = set()

        if agent_type == 'multi':
            moves = agent.get_next_moves()
            for i, move in enumerate(moves):
                if move:
                    robots[i].y, robots[i].x = move
                    robots[i].path.append((robots[i].x, robots[i].y))
        else:
            if not agent.has_completed_path():
                next_pos = agent.get_next_move()
                if next_pos:
                    robots[0].y, robots[0].x = next_pos
                    robots[0].path.append((robots[0].x, robots[0].y))
                    robots[0].isHoldingPackage = agent.is_holding_package

        for i, robot in enumerate(robots):
            robot_rect = pygame.Rect(robot.x * CELL_SIZE, robot.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Use unique color for each robot
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            pygame.draw.rect(win, color, robot_rect)

            # Draw path with transparency, only once per cell
            for x, y in robot.path:
                if (x, y) != (robot.x, robot.y):
                    draw_key = (x, y)
                    if draw_key not in drawn_translucents:
                        pos_x = x * CELL_SIZE + CELL_SIZE // 3
                        pos_y = y * CELL_SIZE + CELL_SIZE // 3
                        win.blit(translucent_surface, (pos_x, pos_y))
                        drawn_translucents.add(draw_key)

            # Draw "1" if holding package
            if agent_type == "multi":
                if agent.robots[i].holding:
                    text = font.render("1", True, BLACK)
                    text_rect = text.get_rect(center=(robot.x * CELL_SIZE + CELL_SIZE // 2, robot.y * CELL_SIZE + CELL_SIZE // 2))
                    win.blit(text, text_rect)
            else:
                if robot.isHoldingPackage:
                    text = font.render("1", True, BLACK)
                    text_rect = text.get_rect(center=(robot.x * CELL_SIZE + CELL_SIZE // 2, robot.y * CELL_SIZE + CELL_SIZE // 2))
                    win.blit(text, text_rect)

        status_text = f"Mode: {agent_type.upper()} | Robots: {len(robots)}"
        status_surface = font.render(status_text, True, BLACK)
        win.blit(status_surface, (10, HEIGHT - 40))

        instr_surface = font.render("Press R to regenerate grid", True, BLACK)
        win.blit(instr_surface, (WIDTH - 300, HEIGHT - 40))

        pygame.display.update()

        if agent_type == 'multi' and agent.all_tasks_done():
            pygame.time.delay(2000)
            return False
        elif agent_type != 'multi' and agent.has_completed_path():
            pygame.time.delay(2000)
            return False

    return False

def main():
    global grid, pickup_locations, dropoff_locations

    parser = argparse.ArgumentParser(description='Warehouse A* Navigation Simulation')
    parser.add_argument('--pickups', type=int, default=1, help='Number of pickup locations')
    parser.add_argument('--dropoffs', type=int, default=1, help='Number of dropoff locations')
    parser.add_argument('--agent', type=str, choices=['single', 'multi'], default='single', help='Choose agent type')
    parser.add_argument('--robots', type=int, default=2, help='Number of robots for multi agent')
    args = parser.parse_args()

    grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)
    while run_simulation(args.agent, args.pickups, args.dropoffs, args.robots):
        grid, pickup_locations, dropoff_locations = generate_random_grid(args.pickups, args.dropoffs)

if __name__ == '__main__':
    main()
