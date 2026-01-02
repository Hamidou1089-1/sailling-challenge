"""
Utility function for saving MyAgent for submission.

This creates a standalone Python file that can be submitted for evaluation.
"""

import os
import numpy as np


def save_my_agent(agent, output_path, agent_class_name="MyAgent"):
    """
    Save a trained MyAgent as a standalone Python file for submission.
    
    Args:
        agent: The trained MyAgent instance
        output_path: Path where to save the agent file (e.g., 'submission/my_agent.py')
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Extract agent parameters
    goal = getattr(agent, 'goal', [16, 31])
    
    # Start building the file content
    file_content = f'''"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a trained Q-learning agent with advanced state features:
- Distance and angle to goal
- Velocity magnitude and wind alignment
- Wind-ahead alignment (checks if wind helps toward goal)
- Wind asymmetry (left vs right wind comparison)
"""

import numpy as np
from agents.base_agent import BaseAgent


class {agent_class_name}(BaseAgent):
    """
    Advanced Q-learning agent with sophisticated state discretization.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Goal position
        self.goal = {goal}
        
        # Caching for performance
        self.weather_cache = None
        self.last_observation_id = None
        self.angle_bins = np.linspace(-np.pi, np.pi, 12)
        
        # Q-table with learned values
        self.q_table = {{}}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
'''
    
    # Add all Q-values (this is the key part - embedding the learned Q-table)
    print(f"Saving {len(agent.q_table)} state-action pairs...")
    for i, (state, values) in enumerate(agent.q_table.items()):
        q_values_str = np.array2string(values, precision=6, separator=', ', max_line_width=100)
        file_content += f"        self.q_table[{state}] = np.array({q_values_str})\n"
        
        # Progress indicator for large Q-tables
        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{len(agent.q_table)} states...")
    
    # Add all the discretization and helper methods
    file_content += '''
    def _discretize_angle(self, angle):
        """Vectorized angle discretization."""
        return np.digitize(angle, self.angle_bins) - 1
    
    def discretize_state(self, observation):
        # Extract base features
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # Cache weather
        obs_id = id(observation)
        if obs_id != self.last_observation_id:
            self.weather_cache = observation[6:].reshape(32, 32, 2)
            self.last_observation_id = obs_id
        weather = self.weather_cache
        
        # Goal features (vectorized)
        goal_vec = np.array(self.goal) - np.array([x, y])
        distance_to_goal = np.linalg.norm(goal_vec)
        angle_to_goal = np.arctan2(goal_vec[1], goal_vec[0])
        
        distance_bin = self.discretize_distance(distance_to_goal)
        angle_to_goal_bin = int((angle_to_goal + np.pi) / (2*np.pi) * 12) % 12
        
        velocity_vec = np.array([vx, vy])

        # Velocity features
        v_magnitude = np.linalg.norm(velocity_vec)
        if v_magnitude < 1.0:
            velocity_magnitude_bin = 0
        elif v_magnitude < 3.0:
            velocity_magnitude_bin = 1
        elif v_magnitude < 6.0:
            velocity_magnitude_bin = 2
        else:
            velocity_magnitude_bin = 3

        if v_magnitude > 0.1 and np.linalg.norm(goal_vec) > 0:
            cos_angle = np.dot(goal_vec, velocity_vec) / (
            np.linalg.norm(goal_vec) * np.linalg.norm(velocity_vec)
            )
            
            # Bin : moving toward goal (1), perpendicular (0), away (-1)
            directness_bin = int((cos_angle + 1) / 2 * 3)  # 0, 1, 2
        else:
            directness_bin = 1
        
        # Wind angle (vectorized)
        v_angle = np.arctan2(vy, vx)
        wind_angle = np.arctan2(wy, wx)
        relative_angle = abs(v_angle - wind_angle)
        if relative_angle > np.pi:
            relative_angle = 2*np.pi - relative_angle
        
        angle_velocity_to_wind_bin = np.digitize(
            relative_angle, 
            [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        ) - 1

        alignment = self.compute_wind_ahead_alignment(
            x, y, angle_to_goal, weather, depth=3
        )


        wind_ahead_bin = self.discretize_wind_ahead(alignment, threshold=0.6)
    
        # Calculer wind_asymmetry
        asymmetry_diff = self.compute_wind_asymmetry(
            x, y, angle_to_goal, weather, depth=3
        )
        wind_asymmetry_bin = self.discretize_asymmetry(asymmetry_diff, threshold=0.4)
        
        return (
            distance_bin,
            angle_to_goal_bin,
            velocity_magnitude_bin,
            angle_velocity_to_wind_bin,
            wind_ahead_bin,
            wind_asymmetry_bin,
            directness_bin
        )
    
    def discretize_distance(self, distance):
        """Discretize distance with 12 fine-grained bins."""
        if distance <= 3:
            return 0
        elif distance <= 6:
            return 1
        elif distance <= 9:
            return 2
        elif distance <= 12:
            return 3
        elif distance <= 15:
            return 4
        elif distance <= 18:
            return 5
        elif distance <= 21:
            return 6
        elif distance <= 24:
            return 7
        elif distance <= 27:
            return 8
        elif distance <= 30:
            return 9
        elif distance <= 35:
            return 10
        else:
            return 11
    
    def compute_wind_ahead_alignment(self, x, y, angle_to_goal, weather, depth=2):
        """Check if wind ahead is aligned with goal direction."""
        goal_dir = np.array([np.cos(angle_to_goal), np.sin(angle_to_goal)])
        
        # Generate all positions to check at once
        positions = np.array([
            [int(x + d * goal_dir[0]), int(y + d * goal_dir[1])]
            for d in range(1, depth + 1)
        ])
        
        # Filter positions in bounds
        valid_mask = (
            (positions[:, 0] >= 0) & (positions[:, 0] < 32) &
            (positions[:, 1] >= 0) & (positions[:, 1] < 32)
        )
        valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            return 0
        
        # Extract winds vectorized
        winds = weather[valid_positions[:, 0], valid_positions[:, 1]]
        avg_wind = winds.mean(axis=0)
        
        # Dot product for alignment
        return np.dot(avg_wind, goal_dir)
    
    def discretize_wind_ahead(self, alignment, threshold=0.5):
        """Discretize wind-ahead alignment into 3 bins."""
        if alignment < -threshold:
            return 0  # unfavorable (headwind)
        elif alignment < threshold:
            return 1  # neutral
        else:
            return 2  # favorable (tailwind)
    
    def compute_wind_asymmetry(self, x, y, angle_to_goal, weather, depth=2):
        """Compare wind strength left vs right of goal direction."""
        angle_left = angle_to_goal + np.pi/2
        angle_right = angle_to_goal - np.pi/2
        
        wind_left = self._get_wind_vectorized(x, y, angle_left, weather, depth)
        wind_right = self._get_wind_vectorized(x, y, angle_right, weather, depth)
        
        return np.linalg.norm(wind_left) - np.linalg.norm(wind_right)
    
    def _get_wind_vectorized(self, x, y, angle, weather, depth):
        """Get average wind in a specific direction (vectorized)."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        positions = np.array([
            [int(x + d * direction[0]), int(y + d * direction[1])]
            for d in range(1, depth + 1)
        ])
        
        valid_mask = (
            (positions[:, 0] >= 0) & (positions[:, 0] < 32) &
            (positions[:, 1] >= 0) & (positions[:, 1] < 32)
        )
        valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            return np.array([0.0, 0.0])
        
        return weather[valid_positions[:, 0], valid_positions[:, 1]].mean(axis=0)
    
    def discretize_asymmetry(self, diff, threshold=0.3):
        """Discretize left-right wind difference."""
        if diff > threshold:
            return 0  # left is better
        elif diff < -threshold:
            return 2  # right is better
        else:
            return 1  # symmetric
    
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use learned Q-values if state is known
        if state not in self.q_table:
            # Default action if state not seen during training
            return 0  # North
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        self.weather_cache = None
        self.last_observation_id = None
    
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
'''
    
    # Write the file
    with open(output_path, 'w') as f:
        f.write(file_content)
    
    print(f"\n✓ Agent saved to {output_path}")
    print(f"✓ The file contains {len(agent.q_table)} state-action pairs")
    print(f"✓ File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"\nYou can now use this file with validate_agent.ipynb and evaluate_agent.ipynb")
    print(f"Make sure the import path is: from agents.base_agent import BaseAgent")



