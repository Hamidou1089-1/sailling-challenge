"""
Utility function for saving DQN agents as standalone files.

Similar to save_qlearning_agent but for DQN:
- Embeds network weights directly in Python code
- Embeds normalization stats
- Includes all physics features computation
- Single file, no external dependencies
"""

import os
import torch
import numpy as np
import pickle


def save_dqn_agent(trainer, output_path, agent_class_name="MyAgentDQN"):
    """
    Save a trained DQN agent as a standalone Python file for submission.
    
    Args:
        trainer: The trained DQNTrainer instance
        output_path: Path where to save the agent file (e.g., 'submission/my_agent.py')
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING DQN AGENT FOR SUBMISSION")
    print(f"{'='*70}")
    
    # Extract network weights
    state_dict = trainer.q_network.state_dict()
    
    # Extract normalization stats
    feature_mean = trainer.feature_mean
    feature_std = trainer.feature_std
    
    # Count parameters
    total_params = sum(p.numel() for p in trainer.q_network.parameters())
    
    print(f"\n📊 Agent Statistics:")
    print(f"   Network parameters: {total_params:,}")
    print(f"   Physics features: {len(feature_mean)}")
    print(f"   Goal position: {trainer.goal}")
    
    # Start building the file content
    file_content = f'''"""
DQN Agent for Sailing Challenge - Trained Model

This file contains a complete trained DQN agent with:
- CNN for wind field encoding (32×32×2 → 64 features)
- Physics-based feature engineering (15 features)
- Embedded network weights ({total_params:,} parameters)
- Embedded normalization statistics
- Compatible with BaseAgent interface

Training configuration:
- Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}
- Gamma: {trainer.gamma}
- Buffer capacity: {trainer.replay_buffer.capacity:,}
- Total training steps: {trainer.steps:,}
- Total episodes: {trainer.episodes:,}
"""

import numpy as np
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent


# =============================================================================
# EMBEDDED NORMALIZATION STATISTICS
# =============================================================================

# These statistics were collected over random episodes for proper normalization
FEATURE_MEAN = np.array({np.array2string(feature_mean, precision=6, separator=', ', max_line_width=100)})

FEATURE_STD = np.array({np.array2string(feature_std, precision=6, separator=', ', max_line_width=100)})


# =============================================================================
# PHYSICS FEATURES COMPUTATION
# =============================================================================

def compute_physics_features(obs, goal=(16, 31)):
    """
    Compute physics-based features from raw observation.
    
    Features (15 total):
    1. distance_to_goal
    2. angle_to_goal
    3. velocity_magnitude
    4. wind_magnitude
    5. directness (velocity-goal alignment)
    6. wind_goal_alignment
    7. relative_wind_angle
    8. predicted_distance
    9. predicted_improvement
    10. wind_ahead_alignment
    11. wind_ahead_magnitude
    12. wind_asymmetry
    13. acceleration_x
    14. acceleration_y
    15. cos(angle_to_goal)
    
    All features are normalized using pre-computed mean and std.
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    wind_field = obs[6:].reshape(32, 32, 2)
    
    # Vectors
    goal_vec = np.array(goal) - np.array([x, y])
    velocity_vec = np.array([vx, vy])
    wind_vec = np.array([wx, wy])
    
    # Magnitudes
    distance_to_goal = np.linalg.norm(goal_vec)
    velocity_magnitude = np.linalg.norm(velocity_vec)
    wind_magnitude = np.linalg.norm(wind_vec)
    
    # Angles
    angle_to_goal = np.arctan2(goal_vec[1], goal_vec[0])
    
    # Directness (cosine similarity velocity-goal)
    if velocity_magnitude > 0.1 and distance_to_goal > 0:
        directness = np.dot(velocity_vec, goal_vec) / (velocity_magnitude * distance_to_goal)
    else:
        directness = 0.0
    
    # Wind-goal alignment
    if wind_magnitude > 0.1 and distance_to_goal > 0:
        wind_goal_alignment = np.dot(wind_vec, goal_vec) / (wind_magnitude * distance_to_goal)
    else:
        wind_goal_alignment = 0.0
    
    # Relative wind angle
    if velocity_magnitude > 0.1 and wind_magnitude > 0.1:
        v_angle = np.arctan2(vy, vx)
        w_angle = np.arctan2(wy, wx)
        relative_wind_angle = v_angle - w_angle
        relative_wind_angle = np.arctan2(np.sin(relative_wind_angle), np.cos(relative_wind_angle))
    else:
        relative_wind_angle = 0.0
    
    # Kinematic prediction
    dt = 1.0
    wind_efficiency = 0.4  # Simplified coefficient
    ax = wind_efficiency * wx
    ay = wind_efficiency * wy
    
    predicted_x = x + vx * dt + 0.5 * ax * dt**2
    predicted_y = y + vy * dt + 0.5 * ay * dt**2
    predicted_goal_vec = np.array(goal) - np.array([predicted_x, predicted_y])
    predicted_distance = np.linalg.norm(predicted_goal_vec)
    predicted_improvement = distance_to_goal - predicted_distance
    
    # Wind ahead (look 3 steps ahead toward goal)
    goal_dir = goal_vec / (distance_to_goal + 1e-6)
    wind_ahead_samples = []
    for d in range(1, 4):
        check_x = int(x + d * goal_dir[0])
        check_y = int(y + d * goal_dir[1])
        if 0 <= check_x < 32 and 0 <= check_y < 32:
            wind_ahead_samples.append(wind_field[check_x, check_y])
    
    if wind_ahead_samples:
        avg_wind_ahead = np.mean(wind_ahead_samples, axis=0)
        wind_ahead_alignment = np.dot(avg_wind_ahead, goal_dir)
        wind_ahead_magnitude = np.linalg.norm(avg_wind_ahead)
    else:
        wind_ahead_alignment = 0.0
        wind_ahead_magnitude = 0.0
    
    # Wind asymmetry (left vs right)
    perp_dir_left = np.array([-goal_dir[1], goal_dir[0]])
    perp_dir_right = np.array([goal_dir[1], -goal_dir[0]])
    
    wind_left_samples = []
    wind_right_samples = []
    for d in range(1, 3):
        check_x = int(x + d * perp_dir_left[0])
        check_y = int(y + d * perp_dir_left[1])
        if 0 <= check_x < 32 and 0 <= check_y < 32:
            wind_left_samples.append(wind_field[check_x, check_y])
        
        check_x = int(x + d * perp_dir_right[0])
        check_y = int(y + d * perp_dir_right[1])
        if 0 <= check_x < 32 and 0 <= check_y < 32:
            wind_right_samples.append(wind_field[check_x, check_y])
    
    if wind_left_samples and wind_right_samples:
        avg_wind_left = np.mean(wind_left_samples, axis=0)
        avg_wind_right = np.mean(wind_right_samples, axis=0)
        wind_asymmetry = np.linalg.norm(avg_wind_left) - np.linalg.norm(avg_wind_right)
    else:
        wind_asymmetry = 0.0
    
    # Assemble raw features
    physics_features_raw = np.array([
        distance_to_goal,
        angle_to_goal,
        velocity_magnitude,
        wind_magnitude,
        directness,
        wind_goal_alignment,
        relative_wind_angle,
        predicted_distance,
        predicted_improvement,
        wind_ahead_alignment,
        wind_ahead_magnitude,
        wind_asymmetry,
        ax,
        ay,
        np.cos(angle_to_goal),
    ], dtype=np.float32)
    
    # Normalize using embedded statistics
    physics_features_norm = (physics_features_raw - FEATURE_MEAN) / (FEATURE_STD + 1e-8)
    
    return physics_features_norm


# =============================================================================
# Q-NETWORK WITH CNN
# =============================================================================

class QNetworkCNN(nn.Module):
    """
    Q-Network avec CNN pour encoder le wind field.
    
    Architecture:
    - CNN: 32x32x2 → 64 features
    - MLP: 15 physics features → 64 features
    - Combine: 128 → 128 → 9 actions
    """
    
    def __init__(self, n_physics_features=15):
        super(QNetworkCNN, self).__init__()
        
        self.wind_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # 32×32 → 16×16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16×16 → 8×8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8×8 → 4×4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4×4×64 → 1×1×64 (GAP)
            nn.Flatten(),  # 64 features (pas de Linear !)
        )
        
        # MLP pour physics features
        self.physics_mlp = nn.Sequential(
            nn.Linear(n_physics_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Réseau combiné
        self.combine = nn.Sequential(
            nn.Linear(128, 128),  # 64 (wind) + 64 (physics)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 9 actions
        )
    
    def forward(self, obs: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Wind field (batch, 2, 32, 32)
            physics: Physics features (batch, 15)
        
        Returns:
            Q-values (batch, 9)
        """
        # Encoder wind field
        wind_features = self.wind_cnn(obs)
        
        # Encoder physics
        physics_features = self.physics_mlp(physics)
        
        # Combiner
        combined = torch.cat([wind_features, physics_features], dim=1)
        q_values = self.combine(combined)
        
        return q_values

# =============================================================================
# AGENT (INHERITS FROM BASEAGENT)
# =============================================================================

class {agent_class_name}(BaseAgent):
    """
    Trained DQN agent for Sailing Challenge.
    Ready for submission and evaluation.
    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.goal = (16, 31)
        
        # Create network
        self.q_network = QNetworkCNN()
        
        # Load embedded weights
        self._load_weights()
        
        # Set to evaluation mode
        self.q_network.eval()
    
    def _load_weights(self):
        """Load network weights from embedded arrays."""
        state_dict = {{}}
'''
    
    # Add network weights
    print(f"\n💾 Embedding network weights...")
    for i, (name, param) in enumerate(state_dict.items()):
        param_np = param.cpu().detach().numpy()
        
        # Convert to string representation
        if param_np.ndim == 1:
            array_str = np.array2string(param_np, precision=6, separator=', ', 
                                       threshold=np.inf, max_line_width=100)
        else:
            array_str = np.array2string(param_np, precision=6, separator=', ',
                                       threshold=np.inf, max_line_width=100)
        
        file_content += f"        state_dict['{name}'] = torch.tensor(\n"
        file_content += f"            np.array({array_str}),\n"
        file_content += f"            dtype=torch.float32\n"
        file_content += f"        )\n"
        
        if (i + 1) % 5 == 0:
            print(f"   {i + 1}/{len(state_dict)} layers embedded")
    
    print(f"   ✓ All {len(state_dict)} layers embedded")
    
    # Add the rest of the agent code
    file_content += '''
        self.q_network.load_state_dict(state_dict)
    
    def act(self, observation):
        """
        Select action with highest Q-value.
        
        Args:
            observation: Raw observation [x, y, vx, vy, wx, wy, wind_field...]
        
        Returns:
            action: Integer between 0 and 8
        """
        # Extract wind field
        wind_field = observation[6:].reshape(32, 32, 2)
        
        # Compute normalized physics features
        physics = compute_physics_features(observation, self.goal)
        
        # Convert to tensors
        wind_tensor = torch.tensor(
            wind_field.transpose(2, 0, 1), 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)
        
        physics_tensor = torch.tensor(
            physics, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def reset(self):
        """Reset agent for new episode."""
        pass  # Nothing to reset for DQN
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
'''
    
    # Write the file
    print(f"\n📝 Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(file_content)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ AGENT SAVED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\n📄 Output file: {output_path}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"🧠 Network parameters: {total_params:,}")
    print(f"🎯 Physics features: {len(feature_mean)}")
    print(f"\n📋 Next steps:")
    print(f"   1. Validate: python src/test_agent_validity.py {output_path}")
    print(f"   2. Evaluate: python src/evaluate_submission.py {output_path} --num-seeds 10")
    print(f"   3. Submit: Upload {output_path} to competition platform")
    print(f"\n{'='*70}\n")


