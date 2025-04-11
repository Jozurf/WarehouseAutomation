from OneRobotAStarAgent import OneRobotAStarAgent
import copy, json, math, heapq, random
from q_learning import QLearningAgent

class RobotAgents:
    """Encapsulates state and path segments for each robot."""
    def __init__(self, start_pos):
        self.position = start_pos
        self.path = []              # Full path for the robot (excluding its starting position)
        self.done = False           # True when the robot finishes its route
        self.holding = False        # True when robot has picked up a package
        self.q_agent = QLearningAgent()
        self.q_agent.load('q_learning_weights.json')  # Load pre-trained weights
        self.last_state = None
        self.last_action = None

class MultiRobotAgent:
    def __init__(self, grid, start_pos, pickups, dropoffs, num_robots):
        self.grid = grid
        self.start_pos = start_pos  # This is the charging station position
        self.pickups = pickups.copy()  # All available pickups
        self.total_pickups = len(pickups)  # Total number of pickups to complete
        self.dropoffs = dropoffs
        self.num_robots = num_robots
        self.robots = [RobotAgents(start_pos) for _ in range(num_robots)]
        self.paths_planned = False
        self.reservation_table = {}  # Maps robots to their currently assigned pickups
        self.completed_pickups = set()  # Track which pickups have been completed
        self.returning_home = set()  # Track which robots are returning to charging station

    def assign_initial_pickups(self):
        """Assign one pickup to each robot initially."""
        available_pickups = self.pickups.copy()
        
        for robot in self.robots:
            if available_pickups:
                # Find the closest pickup to the robot's start position
                closest_pickup = min(available_pickups, 
                                   key=lambda p: self.manhattan(robot.position, p))
                self.reservation_table[robot] = closest_pickup
                available_pickups.remove(closest_pickup)
                
                # Plan path from start to pickup to nearest dropoff
                dropoff = self.find_nearest(closest_pickup, self.dropoffs)
                agent = OneRobotAStarAgent(self.grid, robot.position, closest_pickup, dropoff)
                if agent.plan_path():
                    robot.path = [agent.pickup_path[1:], agent.dropoff_path[1:]]
                else:
                    print(f"Failed to plan initial path for robot")
                    robot.done = True

    def assign_next_pickup(self, robot):
        """Assign the nearest available pickup to the robot after completing a task."""
        # Get all pickups that aren't completed or currently assigned
        available_pickups = [p for p in self.pickups 
                           if p not in self.completed_pickups and 
                           p not in self.reservation_table.values()]
        
        if available_pickups:
            # Find the closest pickup to robot's current position
            closest_pickup = min(available_pickups, 
                               key=lambda p: self.manhattan(robot.position, p))
            self.reservation_table[robot] = closest_pickup
            
            # Plan path to new pickup and nearest dropoff
            dropoff = self.find_nearest(closest_pickup, self.dropoffs)
            agent = OneRobotAStarAgent(self.grid, robot.position, closest_pickup, dropoff)
            if agent.plan_path():
                robot.path = [agent.pickup_path[1:], agent.dropoff_path[1:]]
                return True
        return False

    def update_robot_segments(self, robot, next_pos):
        """Clean up and update pickup, dropoff, and return home segments based on the move."""
        if robot.path and robot.path[0] == []:
            # Remove empty segment
            robot.path.pop(0)
            
            # If we just finished a segment, check if we picked up or dropped off
            x, y = next_pos
            if robot.holding and (x, y) in self.dropoffs:
                robot.holding = False
                print("Robot dropped off the package!")
                # Clear the path to allow getting a new pickup or return home
                robot.path = []
                # Remove the pickup from the reservation table since it's completed
                if robot in self.reservation_table:
                    pickup = self.reservation_table[robot]
                    self.completed_pickups.add(pickup)
                    del self.reservation_table[robot]
                    print(f"Completed pickups: {len(self.completed_pickups)}/{self.total_pickups}")
                    
                    # If all pickups are completed, send robot home
                    if len(self.completed_pickups) == self.total_pickups:
                        self.returning_home.add(robot)
                        agent = OneRobotAStarAgent(self.grid, robot.position, self.start_pos, self.start_pos)
                        if agent.plan_path():
                            robot.path = [agent.pickup_path[1:]]  # Only need path to charging station
                    else:
                        # Try to assign a new pickup
                        if not self.assign_next_pickup(robot):
                            # If no more pickups available, return to charging station
                            self.returning_home.add(robot)
                            agent = OneRobotAStarAgent(self.grid, robot.position, self.start_pos, self.start_pos)
                            if agent.plan_path():
                                robot.path = [agent.pickup_path[1:]]  # Only need path to charging station
                            
            elif not robot.holding and (x, y) == self.start_pos and robot in self.returning_home:
                # Robot has returned to charging station
                robot.done = True
                print("Robot returned to charging station!")
                
            elif not robot.holding and self.grid[x][y] == 4:  # Pickup
                robot.holding = True
                print("Robot picked up the package!")
                # Mark the pickup as completed
                if robot in self.reservation_table:
                    pickup = self.reservation_table[robot]
                    if pickup in self.pickups:
                        self.pickups.remove(pickup)
                    print(f"Completed pickups: {len(self.completed_pickups)}/{self.total_pickups}")
        
        robot.position = next_pos

    def get_next_moves(self):
        """Determine and update the next move for each robot."""
        moves = []
        for robot in self.robots:
            if robot.done or not robot.path:
                moves.append(robot.position)
                continue

            # Get the next position from A* path
            if robot.path[0]:  # If there are points in the current path segment
                next_a_star_pos = robot.path[0][0]  # Next position from A* path
            else:
                next_a_star_pos = robot.position  # Stay in place if segment is empty

            # Check if following A* path would lead to collision
            would_collide = False
            for other_robot in self.robots:
                if other_robot != robot and not other_robot.done:
                    if (abs(next_a_star_pos[0] - other_robot.position[0]) <= 1 and 
                        abs(next_a_star_pos[1] - other_robot.position[1]) <= 1):
                        would_collide = True
                        break

            if would_collide:
                # Use Q-learning for collision avoidance
                current_state = robot.q_agent.get_state(robot, self.grid)
                valid_moves = self.get_valid_moves(robot)
                action = robot.q_agent.get_action(current_state, valid_moves)
                
                if action:
                    dx, dy = action
                    new_x, new_y = robot.position[0] + dx, robot.position[1] + dy
                    next_pos = (new_x, new_y)
                    
                    # Calculate reward based on collision avoidance and distance to goal
                    reward = 0
                    # Negative reward for getting too close to obstacles or other robots
                    adjacent_obstacles = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                                          if self.is_obstacle_or_robot(new_x + dx, new_y + dy))
                    reward -= adjacent_obstacles * 0.5
                    
                    # Positive reward for successful movement
                    reward += 0.1
                    
                    # Extra reward for moving closer to the A* path target
                    if self.manhattan((new_x, new_y), next_a_star_pos) < self.manhattan(robot.position, next_a_star_pos):
                        reward += 0.2
                    
                    # Update Q-values
                    if robot.last_state is not None and robot.last_action is not None:
                        robot.q_agent.update(robot.last_state, robot.last_action, reward, current_state)
                    
                    robot.last_state = current_state
                    robot.last_action = action
                else:
                    next_pos = robot.position
            else:
                # No collision risk, follow A* path
                if self.is_adjacent(robot.position, next_a_star_pos):
                    next_pos = robot.path[0].pop(0)
                else:
                    # Robot deviated from path, get back on track
                    self.get_back_on_track(robot)
                    next_pos = robot.path[0].pop(0) if robot.path[0] else robot.position

            # Update robot state
            self.update_robot_segments(robot, next_pos)
            moves.append(next_pos)

        return moves

    def has_neighbor(self, robot):
        """Check if there are any robots nearby"""
        x, y = robot.position
        for other_robot in self.robots:
            if other_robot != robot and not other_robot.done:
                ox, oy = other_robot.position
                if abs(x - ox) <= 2 and abs(y - oy) <= 2:  # Check 2-block radius
                    return True
        return False

    def get_valid_moves(self, robot):
        """Returns list of valid moves for the robot"""
        possible_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        valid_moves = []
        x, y = robot.position
        for dx, dy in possible_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]):
                if not self.is_obstacle_or_robot(new_x, new_y):
                    valid_moves.append((dx, dy))
        return valid_moves

    def is_obstacle_or_robot(self, x, y):
        """Check if position contains obstacle or another robot"""
        if self.grid[x][y] == 1:  # Check for wall/shelf
            return True
        for robot in self.robots:  # Check for other robots
            if not robot.done and robot.position == (x, y):
                return True
        return False

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
        """Check if all pickups have been completed and all robots have returned to charging station."""
        all_pickups_done = len(self.completed_pickups) == self.total_pickups
        all_robots_home = all(robot.done for robot in self.robots)
        return all_pickups_done and all_robots_home
    
    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
