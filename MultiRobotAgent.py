from OneRobotAStarAgent import OneRobotAStarAgent
import copy

class MultiRobotAgent:
    def __init__(self, grid, start_pos, pickups, dropoffs, num_robots):
        self.grid = grid
        self.start_pos = start_pos
        self.pickups = pickups
        self.dropoffs = dropoffs  # Reusable
        self.num_robots = num_robots
        self.robots = [{
            "position": start_pos,
            "path": [],
            "done": False,
            "holding": False
        } for _ in range(num_robots)]
        self.paths_planned = False
        self.reservation_table = {}  # timestep -> set of reserved positions
        self.edge_table = {}         # timestep -> set of reserved edges

    def assign_tasks(self):
        pickup_tasks = copy.deepcopy(self.pickups)
        assignments = [[] for _ in range(self.num_robots)]
        i = 0
        while pickup_tasks:
            assignments[i % self.num_robots].append(pickup_tasks.pop())
            i += 1
        return assignments

    def plan_paths(self):
        assignments = self.assign_tasks()

        for i, robot in enumerate(self.robots):
            full_path = [self.start_pos]
            current_pos = self.start_pos

            for pickup in assignments[i]:
                nearest_dropoff = self.find_nearest(pickup, self.dropoffs)

                agent = OneRobotAStarAgent(self.grid, current_pos, pickup, nearest_dropoff)
                if not agent.plan_path():
                    print(f"[Error] Robot {i} failed on task {pickup} → {nearest_dropoff}")
                    robot["done"] = True
                    break

                full_path += agent.path[1:]
                current_pos = nearest_dropoff

            if not robot["done"]:
                return_agent = OneRobotAStarAgent(self.grid, current_pos, self.start_pos, self.start_pos)
                if return_agent.plan_path():
                    full_path += return_agent.path[1:]

            robot["path"] = self.resolve_conflicts(i, full_path)

        self.paths_planned = True

    def resolve_conflicts(self, robot_id, path):
        resolved_path = []
        t = 0

        for i in range(len(path)):
            pos = path[i]
            prev = resolved_path[-1] if resolved_path else path[0]

            while (
                pos in self.reservation_table.get(t, set()) or
                (t > 0 and (prev, pos) in self.edge_table.get(t, set()))
            ):
                # Wait at previous position
                resolved_path.append(prev)
                t += 1
                prev = resolved_path[-1]

            resolved_path.append(pos)

            # Reserve vertex
            if t not in self.reservation_table:
                self.reservation_table[t] = set()
            self.reservation_table[t].add(pos)

            # Reserve edge
            if t > 0:
                if t not in self.edge_table:
                    self.edge_table[t] = set()
                self.edge_table[t].add((prev, pos))

            t += 1

        return resolved_path

    def get_next_moves(self):
        moves = []
        for robot in self.robots:
            if robot["done"] or not robot["path"]:
                moves.append(None)
                continue

            next_pos = robot["path"].pop(0)
            robot["position"] = next_pos

            r, c = next_pos
            if self.grid[r][c] == 4:  # Pickup
                robot["holding"] = True
            elif self.grid[r][c] == 5:  # Dropoff
                robot["holding"] = False

            if not robot["path"]:
                robot["done"] = True

            moves.append(next_pos)
        return moves

    def all_tasks_done(self):
        return all(robot["done"] for robot in self.robots)

    def find_nearest(self, current_pos, options):
        return min(options, key=lambda pos: self.manhattan(current_pos, pos))

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
