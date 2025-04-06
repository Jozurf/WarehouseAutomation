import sys
import heapq
import argparse

class OneRobotAStarAgent:
    def __init__(self, grid, start_pos=None, pickup=None, dropoff=None):
        self.grid = grid
        self.start_pos = start_pos
        self.pickup = pickup
        self.dropoff = dropoff
        self.is_holding_package = False
        self.path = []
        self.current_path_index = 0
        self.current_path = []
        self.pickup_path = []
        self.dropoff_path = []
        
    def set_start_position(self, start_pos):
        self.start_pos = start_pos
        
    def set_pickup_dropoff(self, pickup, dropoff):
        self.pickup = pickup
        self.dropoff = dropoff
        
    def plan_path(self):
        """Plan the complete path from start to pickup to dropoff"""
        # First, plan path to pickup
        to_pickup = self.astar_search(self.start_pos, self.pickup)
        if not to_pickup:
            print("No valid path to pickup location!")
            return False
        
        # Then, plan path from pickup to dropoff
        to_dropoff = self.astar_search(self.pickup, self.dropoff)
        if not to_dropoff:
            print("No valid path from pickup to dropoff location!")
            return False
        
        # Combine paths (remove duplicate pickup position)
        self.pickup_path = to_pickup
        self.dropoff_path = to_dropoff
        print("Path to pickup:", self.pickup_path)
        print("Path to dropoff:", self.dropoff_path)
        self.path = to_pickup + to_dropoff[1:]
        self.current_path = self.path
        self.current_path_index = 0
        return True
    
    def astar_search(self, start, goal):
        """A* search algorithm to find the shortest path"""
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
                
                if 0 <= neighbor[0] < len(self.grid) and 0 <= neighbor[1] < len(self.grid[0]) and self.grid[neighbor[0]][neighbor[1]] != 1:
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        
                        # Check if neighbor is already in open_set
                        in_open_set = False
                        for _, (r, c) in open_set:
                            if (r, c) == neighbor:
                                in_open_set = True
                                break
                        
                        if not in_open_set:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def get_next_move(self):
        """Get the next position in the path"""
        if self.current_path_index < len(self.current_path):
            next_pos = self.current_path[self.current_path_index]
            self.current_path_index += 1
            
            # Check if robot is at pickup location
            if not self.is_holding_package and next_pos == self.pickup:
                self.is_holding_package = True
                print("Robot picked up the package!")
                
            # Check if robot is at dropoff location with package
            elif self.is_holding_package and next_pos == self.dropoff:
                self.is_holding_package = False
                print("Package delivered successfully!")
                
            return next_pos
        return None

    def has_completed_path(self):
        """Check if the robot has completed its path"""
        return self.current_path_index >= len(self.current_path)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='A* Pathfinding Agent for Warehouse Navigation')
    parser.add_argument('--pickups', type=int, default=1, help='Number of pickup locations')
    parser.add_argument('--dropoffs', type=int, default=1, help='Number of dropoff locations')
    args = parser.parse_args()
    
    print(f"Starting OneRobotA* Agent with {args.pickups} pickup(s) and {args.dropoffs} dropoff(s)")
    print("Note: This file is meant to be imported by the main simulation. Run the main simulation with these arguments.")
    
if __name__ == "__main__":
    main()