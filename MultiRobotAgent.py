from OneRobotAStarAgent import OneRobotAStarAgent
import copy, json, math, heapq, random
from q_learning import QLearningAgent

class RobotAgents:
    """Encapsulates state and path segments for each robot."""
    def __init__(self, start_pos):
        self.position = start_pos
        self.path = []              # Full path for the robot (excluding its starting position)
        self.done = False           # True when the robot finishes its route
        self.capacity = random.randint(2, 5)  # Random capacity between 2 and 5
        self.current_load = 0       # Current load being carried
        self.holding_packages = []  # List of (position, size) tuples of packages being held
        self.q_agent = QLearningAgent()
        self.q_agent.load('q_learning_weights.json')  # Load pre-trained weights
        self.last_state = None
        self.last_action = None

class MultiRobotAgent:
    def __init__(self, grid, start_pos, pickups, dropoffs, num_robots, package_sizes):
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
        self.package_sizes = package_sizes  # Maps pickup locations to their sizes

    def calculate_package_value(self, package_pos, robot_pos):
        """Calculate the value of a package based on its size, distance, and nearby packages."""
        size = self.package_sizes[package_pos]
        distance = self.manhattan(robot_pos, package_pos)
        
        # Base value: size/distance ratio
        value = size / (distance + 1)  # +1 to avoid division by zero
        
        # Check for nearby packages that could fit
        nearby_bonus = 0
        for other_pickup in self.pickups:
            if other_pickup != package_pos and other_pickup not in self.completed_pickups:
                other_distance = self.manhattan(package_pos, other_pickup)
                if other_distance <= 3:  # Only consider packages within 3 steps
                    nearby_bonus += 0.5  # Bonus for each nearby package
                    
        return value + nearby_bonus

    def get_reserved_pickups(self):
        """Get all pickups that are currently reserved by any robot."""
        reserved = set()
        for pickups in self.reservation_table.values():
            if isinstance(pickups, list):
                reserved.update(pickups)
            else:
                reserved.add(pickups)
        return reserved

    def find_additional_pickups(self, robot, current_pickup, dropoff):
        """Find additional pickups that could be made before going to dropoff."""
        additional_pickups = []
        remaining_capacity = robot.capacity - robot.current_load - self.package_sizes[current_pickup]
        current_pos = current_pickup
        
        # Get all reserved pickups
        reserved_pickups = self.get_reserved_pickups()
        
        # Look for packages that could be picked up efficiently
        while remaining_capacity > 0:
            best_next_pickup = None
            best_value = -1
            
            for pickup in self.pickups:
                if pickup != current_pickup and pickup not in self.completed_pickups \
                   and pickup not in reserved_pickups \
                   and pickup not in additional_pickups:
                    
                    # Check if package would fit
                    if self.package_sizes[pickup] <= remaining_capacity:
                        # Calculate detour cost: extra distance compared to direct path
                        direct_dist = self.manhattan(current_pos, dropoff)
                        detour_dist = self.manhattan(current_pos, pickup) + self.manhattan(pickup, dropoff)
                        detour_cost = detour_dist - direct_dist
                        
                        if detour_cost <= 5:  # Only consider small detours
                            value = self.package_sizes[pickup] / (detour_cost + 1)
                            if value > best_value:
                                best_value = value
                                best_next_pickup = pickup
            
            if best_next_pickup:
                additional_pickups.append(best_next_pickup)
                remaining_capacity -= self.package_sizes[best_next_pickup]
                current_pos = best_next_pickup
            else:
                break
                            
        return additional_pickups

    def update_robot_segments(self, robot, next_pos):
        """Clean up and update pickup, dropoff, and return home segments based on the move."""
        if robot.path and robot.path[0] == []:
            # Remove empty segment
            robot.path.pop(0)
            
            # If we just finished a segment, check if we picked up or dropped off
            x, y = next_pos
            if robot.holding_packages and (x, y) in self.dropoffs:
                # Drop off all packages
                num_packages = len(robot.holding_packages)  # Store count before clearing
                for package_pos, package_size in robot.holding_packages:
                    self.completed_pickups.add(package_pos)
                robot.holding_packages = []
                robot.current_load = 0
                print(f"Robot dropped off {num_packages} packages!")
                
                # Clear the path to allow getting a new pickup or return home
                robot.path = []
                
                # Remove the pickups from the reservation table
                if robot in self.reservation_table:
                    pickups = self.reservation_table[robot]
                    del self.reservation_table[robot]
                    print(f"Completed pickups: {len(self.completed_pickups)}/{self.total_pickups}")
                    
                    # If all pickups are completed, send robot home
                    if len(self.completed_pickups) == self.total_pickups:
                        self.returning_home.add(robot)
                        agent = OneRobotAStarAgent(self.grid, robot.position, self.start_pos, self.start_pos)
                        if agent.plan_path():
                            robot.path = [agent.pickup_path[1:]]
                    else:
                        # Try to assign a new pickup
                        if not self.assign_next_pickup(robot):
                            # If no more pickups available, return to charging station
                            self.returning_home.add(robot)
                            agent = OneRobotAStarAgent(self.grid, robot.position, self.start_pos, self.start_pos)
                            if agent.plan_path():
                                robot.path = [agent.pickup_path[1:]]
                            
            elif not robot.holding_packages and (x, y) == self.start_pos and robot in self.returning_home:
                # Robot has returned to charging station
                robot.done = True
                print("Robot returned to charging station!")
                
            elif self.grid[x][y] == 4:  # Pickup
                if (x, y) in self.package_sizes and robot in self.reservation_table:
                    pickups = self.reservation_table[robot]
                    if (x, y) in pickups:  # Only pick up if it's in our planned pickups
                        package_size = self.package_sizes[(x, y)]
                        if package_size <= (robot.capacity - robot.current_load):
                            robot.holding_packages.append(((x, y), package_size))
                            robot.current_load += package_size
                            print(f"Robot picked up package of size {package_size}! Current load: {robot.current_load}/{robot.capacity}")
                            # Mark the pickup as in progress
                            if (x, y) in self.pickups:
                                self.pickups.remove((x, y))
        
        robot.position = next_pos

    def assign_initial_pickups(self):
        """Assign initial pickups to robots based on value and capacity."""
        available_pickups = self.pickups.copy()
        
        for robot in self.robots:
            if available_pickups:
                # Calculate value for all pickups that fit
                pickup_values = []
                for pickup in available_pickups:
                    if self.package_sizes[pickup] <= robot.capacity:
                        value = self.calculate_package_value(pickup, robot.position)
                        pickup_values.append((pickup, value))
                
                if pickup_values:
                    # Sort by value and take the best one
                    pickup_values.sort(key=lambda x: x[1], reverse=True)
                    best_pickup = pickup_values[0][0]
                    
                    # Find nearest dropoff
                    dropoff = self.find_nearest(best_pickup, self.dropoffs)
                    
                    # Plan initial path to best pickup
                    agent = OneRobotAStarAgent(self.grid, robot.position, best_pickup, dropoff)
                    if agent.plan_path():
                        # Look for additional pickups
                        additional_pickups = self.find_additional_pickups(robot, best_pickup, dropoff)
                        
                        # Build complete path including all pickups
                        complete_path = []
                        current_pos = robot.position
                        all_pickups = [best_pickup] + additional_pickups
                        
                        # Path to first pickup
                        agent = OneRobotAStarAgent(self.grid, current_pos, best_pickup, dropoff)
                        if agent.plan_path():
                            complete_path.append(agent.pickup_path[1:])
                            current_pos = best_pickup
                            
                            # Paths to additional pickups
                            for pickup in additional_pickups:
                                agent = OneRobotAStarAgent(self.grid, current_pos, pickup, dropoff)
                                if agent.plan_path():
                                    complete_path.append(agent.pickup_path[1:])
                                    current_pos = pickup
                            
                            # Final path to dropoff
                            agent = OneRobotAStarAgent(self.grid, current_pos, dropoff, dropoff)
                            if agent.plan_path():
                                complete_path.append(agent.pickup_path[1:])
                                
                                robot.path = complete_path
                                self.reservation_table[robot] = all_pickups
                                
                                # Remove assigned pickups from available ones
                                for pickup in all_pickups:
                                    if pickup in available_pickups:
                                        available_pickups.remove(pickup)
                    else:
                        print(f"Failed to plan initial path for robot")
                        robot.done = True

    def assign_next_pickup(self, robot):
        """Assign next pickups to robot after completing a delivery."""
        reserved_pickups = self.get_reserved_pickups()
        available_pickups = [p for p in self.pickups 
                           if p not in self.completed_pickups and 
                           p not in reserved_pickups]
        
        if available_pickups:
            # Calculate value for all pickups that would fit
            pickup_values = []
            for pickup in available_pickups:
                if self.package_sizes[pickup] <= robot.capacity:
                    value = self.calculate_package_value(pickup, robot.position)
                    pickup_values.append((pickup, value))
            
            if pickup_values:
                # Sort by value and take the best one
                pickup_values.sort(key=lambda x: x[1], reverse=True)
                best_pickup = pickup_values[0][0]
                
                # Find nearest dropoff
                dropoff = self.find_nearest(best_pickup, self.dropoffs)
                
                # Plan path to best pickup
                agent = OneRobotAStarAgent(self.grid, robot.position, best_pickup, dropoff)
                if agent.plan_path():
                    # Look for additional pickups
                    additional_pickups = self.find_additional_pickups(robot, best_pickup, dropoff)
                    
                    # Build complete path including all pickups
                    complete_path = []
                    current_pos = robot.position
                    all_pickups = [best_pickup] + additional_pickups
                    
                    # Path to first pickup
                    agent = OneRobotAStarAgent(self.grid, current_pos, best_pickup, dropoff)
                    if agent.plan_path():
                        complete_path.append(agent.pickup_path[1:])
                        current_pos = best_pickup
                        
                        # Paths to additional pickups
                        for pickup in additional_pickups:
                            agent = OneRobotAStarAgent(self.grid, current_pos, pickup, dropoff)
                            if agent.plan_path():
                                complete_path.append(agent.pickup_path[1:])
                                current_pos = pickup
                        
                        # Final path to dropoff
                        agent = OneRobotAStarAgent(self.grid, current_pos, dropoff, dropoff)
                        if agent.plan_path():
                            complete_path.append(agent.pickup_path[1:])
                            
                            robot.path = complete_path
                            self.reservation_table[robot] = all_pickups
                            return True
        return False

    def get_next_moves(self):
        """Determine and update the next move for each robot."""
        moves = []
        for robot in self.robots:
            if robot.done:
                moves.append(robot.position)
                continue

            # If robot has no path, try to assign a new pickup
            if not robot.path:
                if not self.assign_next_pickup(robot):
                    # If no pickups available and robot has no packages, send it home
                    if not robot.holding_packages:
                        self.returning_home.add(robot)
                        agent = OneRobotAStarAgent(self.grid, robot.position, self.start_pos, self.start_pos)
                        if agent.plan_path():
                            robot.path = [agent.pickup_path[1:]]
                        else:
                            robot.done = True
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

    def find_nearest(self, pos, targets):
        """Find the nearest target position to the given position."""
        return min(targets, key=lambda p: self.manhattan(pos, p))
    
    def all_tasks_done(self):
        """Check if all pickups have been completed and all robots have returned to charging station."""
        all_pickups_done = len(self.completed_pickups) == self.total_pickups
        all_robots_home = all(robot.done for robot in self.robots)
        return all_pickups_done and all_robots_home
    
    def manhattan(self, pos1, pos2):
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
