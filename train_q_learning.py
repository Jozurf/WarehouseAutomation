import random
from robot import Robot
import numpy as np

def generate_small_grid(size=10):
    """Generate a smaller grid for faster training"""
    grid = [[0 for _ in range(size)] for _ in range(size)]
    
    # Add walls around the edges
    for i in range(size):
        grid[0][i] = grid[size-1][i] = grid[i][0] = grid[i][size-1] = 1
    
    # Add some random obstacles (15% chance)
    for i in range(2, size-2):
        for j in range(2, size-2):
            if random.random() < 0.15:
                grid[i][j] = 1
                
    return grid

def find_random_valid_position(grid):
    """Find a random valid position on the grid"""
    while True:
        x = random.randint(1, len(grid[0]) - 2)
        y = random.randint(1, len(grid) - 2)
        if grid[y][x] == 0:  # Empty cell
            return x, y

def train_q_learning(episodes=10000, steps_per_episode=100, num_robots=3, grid_size=8):
    """Train multiple Q-learning agents for collision avoidance"""
    # Training metrics
    episode_rewards = []
    collision_counts = []
    
    for episode in range(episodes):
        # Generate new small grid for each episode
        grid = generate_small_grid(grid_size)
        
        # Create multiple robots at different positions
        robots = []
        for _ in range(num_robots):
            x, y = find_random_valid_position(grid)
            robot = Robot(x, y, grid)
            # Reduce epsilon over time for exploration-exploitation trade-off
            robot.q_agent.epsilon = max(0.01, 0.5 - episode / episodes)
            robots.append(robot)
        
        # create a goal for each robot to go to each episode once reached will be replaced
        for robot in robots:
            goal_x, goal_y = find_random_valid_position(grid)
            robot.waypoint = [[(goal_x, goal_y)]]
        
        episode_reward = 0
        collisions = 0
        
        for step in range(steps_per_episode):
            # Update temporary grid with robot positions for collision detection
            temp_grid = [row[:] for row in grid]
            for r in robots:
                temp_grid[r.y][r.x] = 2  # Mark robot positions as obstacles
            
            # Move each robot and track rewards
            for robot in robots:
                # Update robot's grid view to include other robots
                robot.grid = temp_grid
                old_pos = (robot.x, robot.y)
                robot.move()

                if not robot.waypoint:
                    new_waypoint = find_random_valid_position(grid)
                    robot.waypoint = [[new_waypoint]]
                # Check for collisions
                if (robot.x, robot.y) == old_pos:  # Robot couldn't move due to collision
                    collisions += 1
                    episode_reward -= 1.0  # Penalty for collision
                else:
                    episode_reward += 0.1  # Small reward for successful movement
        
        # Track metrics
        episode_rewards.append(episode_reward / steps_per_episode)
        collision_counts.append(collisions)
        
        # Save weights and print progress periodically
        if (episode + 1) % 1000 == 0:
            robots[0].q_agent.save('q_learning_weights.json')  # Save weights from first robot
            print(f"Episode {episode + 1}")
            print(f"Average Reward: {np.mean(episode_rewards[-1000:]):.2f}")
            print(f"Average Collisions: {np.mean(collision_counts[-1000:]):.2f}")
    
    # Save final weights
    robots[0].q_agent.save('q_learning_weights.json')  # Save weights from first robot
    print("\nTraining completed!")
    print(f"Final Average Reward: {np.mean(episode_rewards[-1000:]):.2f}")
    print(f"Final Average Collisions: {np.mean(collision_counts[-1000:]):.2f}")

if __name__ == "__main__":
    train_q_learning() 