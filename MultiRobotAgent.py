from OneRobotAStarAgent import OneRobotAStarAgent
import copy, json, math, heapq, random

class RobotAgents:
    """Encapsulates state and path segments for each robot."""
    def __init__(self, start_pos):
        self.position = start_pos
        self.path = []              # Full path for the robot (excluding its starting position)
        self.done = False           # True when the robot finishes its route
        self.holding = False        # True when robot has picked up a package

class MultiRobotAgent:
    def __init__(self, grid, start_pos, pickups, dropoffs, num_robots):
        self.grid = grid
        with open("weights.json") as f:
            self.weights = json.load(f)
        self.start_pos = start_pos
        self.pickups = pickups
        self.dropoffs = dropoffs  # Dropoff points (reusable)
        self.num_robots = num_robots
        self.robots = [RobotAgents(start_pos) for _ in range(num_robots)]
        self.paths_planned = False
        self.reservation_table = {}  # Not used yet
        self.edge_table = {}         # Not used yet

    def assign_tasks(self):
        """Assign pickup tasks to robots in a round-robin fashion."""
        tasks = copy.deepcopy(self.pickups)
        assignments = [[] for _ in range(self.num_robots)]
        for i, task in enumerate(tasks):
            assignments[i % self.num_robots].append(task)
        return assignments

    def plan_paths(self):
        """Plan full paths for all robots including pickups, dropoffs, and return home."""
        assignments = self.assign_tasks()

        for i, robot in enumerate(self.robots):
            full_path = []
            current_pos = self.start_pos

            for pickup in assignments[i]:
                dropoff = self.find_nearest(pickup, self.dropoffs)
                agent = OneRobotAStarAgent(self.grid, current_pos, pickup, dropoff)
                if not agent.plan_path():
                    print(f"[Error] Robot {i} failed on task {pickup} -> {dropoff}")
                    robot.done = True
                    break

                # Append segment paths while excluding duplicate starting positions.
                full_path.append(agent.pickup_path[1:])
                full_path.append(agent.dropoff_path[1:])
                print(f"Robot {i} assigned path: {agent.pickup_path[1:]} -> {agent.dropoff_path[1:]}")
                current_pos = dropoff

            if not robot.done:
                # Plan path for returning home.
                return_agent = OneRobotAStarAgent(self.grid, current_pos, self.start_pos, self.start_pos)
                if return_agent.plan_path():
                    full_path.append(return_agent.pickup_path[1:])
                else:
                    print(f"[Error] Robot {i} failed to plan return home.")
                    robot.done = True

            # Assign planned paths to the robot.
            robot.path = full_path

            # Debug printouts.
            print(f"Robot {i} assigned full path: {robot.path}")
            print("Robot starting at:", robot.position)

        self.paths_planned = True

    def get_next_moves(self):
        """Determine and update the next move for each robot."""
        moves = []
        for robot in self.robots:
            if robot.done or not robot.path:
                moves.append(None)
                continue

            # Occasionally deviate from the planned path.
            if self.has_neighbor(robot):
                next_pos = self.get_best_action(robot)
            else:
                # If the robot is near the expected next position, follow the planned path.
                if self.is_adjacent(robot.position, robot.path[0][0]):
                    next_pos = robot.path[0].pop(0)
                else:
                    print("Robot is off track:", robot.position, "expected next:", robot.path[0][0])
                    self.get_back_on_track(robot)
                    next_pos = robot.path[0].pop(0)

            # Update holding state, path of the robot and position.
            self.update_robot_segments(robot, next_pos)

            if not robot.path:
                robot.done = True

            moves.append(next_pos)
        return moves

    def update_robot_segments(self, robot, next_pos):
        """Clean up and update pickup, dropoff, and return home segments based on the move."""
        if robot.path[0] == []:
            # check 4 -> pickup, 5 -> dropoff
            x, y = next_pos
            if robot.holding and self.grid[x][y] == 5:
                robot.holding = False
                print("Robot dropped off the package!")
            elif not robot.holding and self.grid[x][y] == 4:
                robot.holding = True
                print("Robot picked up the package!")
            robot.path.pop(0)  # Remove empty segments
        
        robot.position = next_pos  
        



    def has_neighbor(self, robot):
        """Randomly decide (20% chance) to simulate deviation due to a nearby robot."""
        if random.random() < 0.2:
            return True
        return False

    def get_best_action(self, robot):
        """Return a random valid adjacent move from the robot's current position."""
        actions = []
        x, y = robot.position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]) and self.grid[nx][ny] != 1:
                actions.append((nx, ny))
        return random.choice(actions) if actions else robot.position

    def is_adjacent(self, pos, target):
        """Check if target is one block away from the current position."""
        x, y = pos
        a, b = target
        return (x == a and abs(y - b) == 1) or (y == b and abs(x - a) == 1)

    def get_back_on_track(self, robot):
        """Replan the path segment when a robot deviates from its original path."""
        # Determine which segment the robot should follow.
        segment = robot.path[0]

        # Find the closest point in the segment to the current position.
        closest_point = min(segment, key=lambda p: math.hypot(p[0] - robot.position[0],
                                                               p[1] - robot.position[1]))
        index_in_segment = segment.index(closest_point)

        # Replan path from current position to the closest point.
        path_to_closest = self.a_star(self.grid, robot.position, closest_point)
        if not path_to_closest:
            print("A* failed to find a path to get back on track.", robot.position, closest_point, segment)
            return

        # Merge the new path with the remainder of the segment.
        new_segment = path_to_closest[1:-1] + segment[index_in_segment:]
        
        # Update the corresponding segment.
        robot.path[0] = new_segment

    def a_star(self, grid, start, goal):
        """A simple A* algorithm to compute a path from start to goal."""
        rows, cols = len(grid), len(grid[0])
        def heuristic(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1])
        
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] in [0, 4, 5]:  # Walkable cells
                    neighbor = (nx, ny)
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        return []  # Return empty if no path is found

    def find_nearest(self, current_pos, options):
        """Return the option that is closest to the current position (using Manhattan distance)."""
        return min(options, key=lambda pos: self.manhattan(current_pos, pos))
    
    def all_tasks_done(self):
        """Check if all robots have completed their tasks."""
        return all(robot.done for robot in self.robots)
    
    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
