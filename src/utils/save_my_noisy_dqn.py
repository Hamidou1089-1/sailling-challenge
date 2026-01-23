"""
Utility to save DQN agents for Codabench submission (Pure NumPy - No PyTorch).


"""

import os
import torch
import numpy as np


def array_to_string(arr, name="array"):
    """
    Convertit un array NumPy en string Python optimisé.
    
    Utilise np.array2string avec des paramètres optimisés pour la lisibilité.
    """
    if arr.ndim == 1:
        # Vecteur 1D
        return np.array2string(arr, separator=', ', max_line_width=100, 
                              threshold=np.inf, precision=6)
    else:
        # Tenseur multi-dim
        return np.array2string(arr, separator=', ', max_line_width=100,
                              threshold=np.inf, precision=6)


def save_dqn_agent(trainer, output_path, agent_class_name="SubmissionAgent"):
    """
    Save a DQN agent with NoisyNet for Codabench submission (Pure NumPy - FAST VERSION).
    
    Cette version écrit les poids directement dans le code (pas de base64/pickle).
    Beaucoup plus rapide à l'inférence.
    
    Args:
        trainer: The trained DQNTrainer instance with q_network (CNN + MLP + NoisyNet)
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING DQN AGENT WITH NOISYNET FOR CODABENCH (FAST VERSION)")
    print(f"{'='*70}")
    
    # Extract all network weights from PyTorch model
    model = trainer.q_network
    weights_dict = {}
    
    print("\n📦 Extracting NoisyNet weights (mu only for inference)...")
    for name, param in model.state_dict().items():
        # Pour NoisyNet: on garde seulement weight_mu et bias_mu
        # On ignore weight_sigma, bias_sigma, weight_epsilon, bias_epsilon
        if 'epsilon' in name or 'sigma' in name:
            print(f"   ⊘ {name}: SKIPPED (noise parameter, not needed for inference)")
            continue
        
        # Renomme weight_mu -> weight, bias_mu -> bias pour simplifier
        clean_name = name.replace('_mu', '')
        weights_dict[clean_name] = param.cpu().detach().numpy()
        print(f"   ✓ {clean_name}: {weights_dict[clean_name].shape}")
    
    # Get configuration
    goal = trainer.goal
    n_physics_features = 12
    total_params = sum(w.size for w in weights_dict.values())
    
    print(f"\n📊 Agent Configuration:")
    print(f"   Goal position: {goal}")
    print(f"   Physics features: {n_physics_features}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   NoisyNet: Enabled (inference uses mu only)")
    
    # Create the submission file content - HEADER
    file_content = f'''"""
Standalone DQN Agent with NoisyNet for Sailing Challenge (Pure NumPy - FAST VERSION)

⚡ POIDS EN DUR POUR INFÉRENCE RAPIDE
Pas de décodage base64/pickle → Temps d'inférence minimal

This agent uses a CNN + MLP architecture with NoisyNet implemented entirely in NumPy:
- CNN: 32x32x2 wind field → 64 features
- MLP: {n_physics_features} physics features → 64 features (trained with NoisyNet)
- Combined: 128 → 128 → 9 Q-values (trained with NoisyNet)

For inference, only the mu (mean) parameters are used (no noise).
No external dependencies required (beyond NumPy and standard library).

Total parameters: {total_params:,}
"""

import numpy as np
from typing import Tuple


from evaluator.base_agent import BaseAgent


def compute_physics_features(obs: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """
    Compute normalized physics-based features from observation.
    
    Args:
        obs: Raw observation [x, y, vx, vy, wx, wy, ...]
        goal: Goal position (x, y)
    
    Returns:
        Physics features array ({n_physics_features} features)
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]  # Local wind
    
    # Physical constants
    MAX_DIST = 45.0
    MAX_SPEED = 5.0
    MAX_WIND = 5.0
    
    # Position relative to goal
    dx = goal[0] - x
    dy = goal[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > 0:
        dir_goal_x = dx / dist
        dir_goal_y = dy / dist
    else:
        dir_goal_x, dir_goal_y = 0.0, 0.0
    
    feat_dist = dist / MAX_DIST
    
    # Boat velocity
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.001:
        dir_boat_x = vx / speed
        dir_boat_y = vy / speed
    else:
        dir_boat_x, dir_boat_y = 0.0, 0.0
    
    feat_speed = speed / MAX_SPEED
    
    # Wind
    wind_speed = np.sqrt(wx**2 + wy**2)
    if wind_speed > 0:
        dir_wind_x = wx / wind_speed
        dir_wind_y = wy / wind_speed
    else:
        dir_wind_x, dir_wind_y = 0.0, 0.0
    
    feat_wind_str = wind_speed / MAX_WIND
    
    # Dot products (alignments)
    feat_align_goal = dir_boat_x * dir_goal_x + dir_boat_y * dir_goal_y
    feat_wind_goal = dir_wind_x * dir_goal_x + dir_wind_y * dir_goal_y
    feat_angle_wind = dir_boat_x * dir_wind_x + dir_boat_y * dir_wind_y
    
    # Cross products (lateral positioning)
    feat_cross_goal = dir_boat_x * dir_goal_y - dir_boat_y * dir_goal_x
    feat_cross_wind = dir_boat_x * dir_wind_y - dir_boat_y * dir_wind_x
    
    features = np.array([
        feat_dist,
        feat_speed,
        feat_wind_str,
        feat_align_goal,
        feat_wind_goal,
        feat_angle_wind,
        feat_cross_goal,
        feat_cross_wind,
        dir_boat_x,
        dir_boat_y,
        dir_wind_x,
        dir_wind_y
    ], dtype=np.float32)
    
    return features


class NumpyCNN:
    """
    CNN implemented in pure NumPy for wind field encoding.
    
    Architecture:
    - Conv2d(2, 16, kernel=5, stride=2, padding=2): 32x32 → 16x16
    - ReLU
    - Conv2d(16, 32, kernel=3, stride=2, padding=1): 16x16 → 8x8
    - ReLU
    - Conv2d(32, 64, kernel=3, stride=2, padding=1): 8x8 → 4x4
    - ReLU
    - AdaptiveAvgPool2d(1): 4x4 → 1x1 (Global Average Pooling)
    - Flatten: 64 features
    """
    
    def __init__(self, weights_dict):
        self.weights = weights_dict
        
    @staticmethod
    def conv2d(x, weight, bias, stride=1, padding=0):
        """
        Manual 2D convolution implementation.
        
        Args:
            x: Input (batch, in_channels, H, W)
            weight: Filters (out_channels, in_channels, kH, kW)
            bias: Bias (out_channels,)
            stride: Stride
            padding: Padding
        
        Returns:
            Output (batch, out_channels, H_out, W_out)
        """
        batch_size, in_channels, H, W = x.shape
        out_channels, _, kH, kW = weight.shape
        
        # Apply padding
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
            H, W = H + 2 * padding, W + 2 * padding
        
        # Calculate output dimensions
        H_out = (H - kH) // stride + 1
        W_out = (W - kW) // stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_channels, H_out, W_out), dtype=np.float32)
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        receptive_field = x[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, oc, i, j] = np.sum(receptive_field * weight[oc]) + bias[oc]
        
        return output
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def adaptive_avg_pool2d(x):
        """Global average pooling: (B, C, H, W) → (B, C, 1, 1)"""
        return np.mean(x, axis=(2, 3), keepdims=True)
    
    def forward(self, x):
        """
        Forward pass through CNN.
        
        Args:
            x: Wind field (batch, 2, 32, 32)
        
        Returns:
            Features (batch, 64)
        """
        # Conv1: 32x32x2 → 16x16x16
        x = self.conv2d(x, 
                       self.weights['wind_cnn.0.weight'],
                       self.weights['wind_cnn.0.bias'],
                       stride=2, padding=2)
        x = self.relu(x)
        
        # Conv2: 16x16x16 → 8x8x32
        x = self.conv2d(x,
                       self.weights['wind_cnn.2.weight'],
                       self.weights['wind_cnn.2.bias'],
                       stride=2, padding=1)
        x = self.relu(x)
        
        # Conv3: 8x8x32 → 4x4x64
        x = self.conv2d(x,
                       self.weights['wind_cnn.4.weight'],
                       self.weights['wind_cnn.4.bias'],
                       stride=2, padding=1)
        x = self.relu(x)
        
        # Global Average Pooling: 4x4x64 → 1x1x64
        x = self.adaptive_avg_pool2d(x)
        
        # Flatten: (batch, 64, 1, 1) → (batch, 64)
        x = x.reshape(x.shape[0], -1)
        
        return x


class NumpyMLP:
    """
    Multi-layer perceptron implemented in pure NumPy.
    For NoisyNet trained models, uses only mu (mean) parameters for inference.
    """
    
    def __init__(self, weights_dict):
        self.weights = weights_dict
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    def forward(self, x, prefix):
        """
        Forward pass through MLP layers.
        
        Args:
            x: Input features
            prefix: Weight prefix ('physics_mlp' or 'combine')
        
        Returns:
            Output features
        """
        layer_idx = 0
        while f'{{prefix}}.{{layer_idx}}.weight' in self.weights:
            weight = self.weights[f'{{prefix}}.{{layer_idx}}.weight']
            bias = self.weights[f'{{prefix}}.{{layer_idx}}.bias']
            
            # Linear layer: x @ W.T + b
            x = np.dot(x, weight.T) + bias
            
            # Check if there's another layer after this one
            next_layer_exists = f'{{prefix}}.{{layer_idx + 2}}.weight' in self.weights
            
            # Apply ReLU only if not the last layer
            if next_layer_exists or (prefix != 'combine'):
                x = self.relu(x)
            
            layer_idx += 2  # Skip ReLU layer index
        
        return x


# =============================================================================
# POIDS DU MODÈLE (EN DUR - PAS DE DÉCODAGE)
# =============================================================================

def load_weights():
    """Charge les poids (directement depuis le code, ultra-rapide)."""
    w = {{}}
'''
    
    # Écrire les poids directement
    print(f"\n💾 Writing weights directly in code (this may take 1-2 minutes)...")
    
    for i, (name, arr) in enumerate(weights_dict.items(), 1):
        print(f"   [{i}/{len(weights_dict)}] {name}...", end='', flush=True)
        
        # Convertir en string
        arr_str = array_to_string(arr, name)
        
        # Ajouter au fichier
        file_content += f"    w['{name}'] = np.array({arr_str}, dtype=np.float32)"
        
        # Reshape si nécessaire
        if arr.ndim > 1:
            file_content += f".reshape{arr.shape}"
        
        file_content += "\n"
        print(" ✓")
    
    # Fin de la fonction load_weights
    file_content += "    return w\n\n"
    
    # Classe de l'agent
    file_content += f'''
class MyAgent(BaseAgent):
    
    
    def __init__(self):
        super().__init__()
        
        # Goal position
        self.goal = {goal}
        
        # Load weights (instantané!)
        weights = load_weights()
        
        # Initialize networks
        self.cnn = NumpyCNN(weights)
        self.mlp = NumpyMLP(weights)
    
    def act(self, observation):
        """
        Select action with highest Q-value.
        
        Args:
            observation: Raw observation [x, y, vx, vy, wx, wy, wind_field...]
        
        Returns:
            action: Integer between 0 and 8
        """
        # Extract wind field (last 2048 values = 32x32x2)
        wind_field = observation[6:].reshape(32, 32, 2)
        
        # Transpose to (2, 32, 32) and add batch dimension
        wind_tensor = wind_field.transpose(2, 0, 1).astype(np.float32)
        wind_tensor = wind_tensor.reshape(1, 2, 32, 32)
        
        # Compute physics features
        physics = compute_physics_features(observation, self.goal)
        physics_tensor = physics.reshape(1, -1)
        
        # Forward pass through CNN
        wind_features = self.cnn.forward(wind_tensor)
        
        # Forward pass through physics MLP
        physics_features = self.mlp.forward(physics_tensor, 'physics_mlp')
        
        # Combine features
        combined = np.concatenate([wind_features, physics_features], axis=1)
        
        # Forward pass through combine network
        q_values = self.mlp.forward(combined, 'combine')
        
        # Select action with highest Q-value
        action = np.argmax(q_values[0])
        
        return int(action)
    
    def reset(self):
        """Reset agent for new episode."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
'''
    
    # Write the file
    print(f"\n📝 Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ AGENT SAVED SUCCESSFULLY FOR CODABENCH (FAST VERSION)")
    print(f"{'='*70}")
    print(f"\n📄 Output file: {output_path}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"🧠 Total parameters: {total_params:,}")
    print(f"🎯 Physics features: {n_physics_features}")
    print(f"\n✨ Key features:")
    print(f"   ✓ Pure NumPy implementation (no PyTorch)")
    print(f"   ✓ Single file (no external dependencies)")
    print(f"   ✓ CNN + MLP architecture preserved")
    print(f"   ✓ NoisyNet training (inference uses mu only)")
    print(f"   ✓ Weights hardcoded in file (NO base64/pickle decoding)")
    print(f"   ✓ FAST loading and inference (~10x faster)")
    print(f"\n📋 Next steps:")
    print(f"   1. Test locally: python {output_path}")
    print(f"   2. Validate: python src/test_agent_validity.py {output_path}")
    print(f"   3. Submit to Codabench")
    print(f"\n{'='*70}\n")