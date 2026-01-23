"""
Attention-Based DQN Agent for Sailing Challenge (ViT-style)

Architecture:
- Wind Field → Patches → Tokens (comme Vision Transformer)
- Physics → Queries
- Cross-Attention: Queries attend aux Wind Tokens
- Dueling Head: Q = V + A - mean(A)

L'agent "regarde" les zones pertinentes du wind field selon son état.
"""

import numpy as np
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_agent import BaseAgent
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from wind_scenarios.env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario


# =============================================================================
# CURRICULUM LEARNING
# =============================================================================
def generate_curriculum_params(progress):
    """
    Générateur d'environnement robuste pour généralisation.
    
    Args:
        progress (float): 0.0 (Début) -> 1.0 (Fin)
    """
    
    # --- 1. DIRECTION : Toujours Aléatoire 360° ---
    # C'est CRUCIAL. L'agent doit comprendre que le vent peut venir de n'importe où,
    # même à l'épisode 1. La facilité vient de la stabilité, pas de la direction.
    theta = np.random.uniform(0, 2 * np.pi)
    wind_dir = (np.cos(theta), np.sin(theta))
    wind_scen_names = ['static_headwind', 'training_1', 'training_3', 'simple_static']
    # --- 2. GESTION DE LA DIFFICULTÉ (Le "Recall") ---
    # On garde 20-30% d'épisodes "Faciles" (Vent stable) tout le temps.
    # Cela sert d'ancrage pour que l'agent n'oublie pas les bases.
    if np.random.random() < 0.4:
        wind_scen = get_wind_scenario(np.random.choice(wind_scen_names))
        wind_init_params = wind_scen['wind_init_params']
        wind_evol_params = wind_scen['wind_evol_params']
        return wind_init_params, wind_evol_params
    else:
       
        # On ajoute un petit bruit pour ne pas être trop linéaire.
        difficulty = 0.5 + np.random.uniform(-0.4, 0.45)

    # --- 3. PARAMÈTRES DU VENT ---
    
    # Vitesse : 3.0 est la vitesse standard. 
    # Plus c'est dur, plus on s'éloigne de cette norme (vent très faible ou tempête).
    # difficulty 0 -> speed 3.0
    # difficulty 1 -> speed entre 1.0 et 5.0
    base_speed = 3.0 + np.random.uniform(-0.01, 0.01)
    
    wind_init_params = {
        'base_speed': base_speed,
        'base_direction': wind_dir,
        
        # Echelle : 128 (Large/Facile) -> 16 (Haché/Dur)
        'pattern_scale': np.random.choice([32, 32, 64, 128, 128]), 
        
        # Force des turbulences
        'pattern_strength': np.clip(0.08 + (0.75 * difficulty), 0.1, 0.62),
        'strength_variation': np.clip(0.08 + (0.5 * difficulty), 0.1, 0.42),
        'noise': np.clip(0.08 + (0.15 * difficulty), 0, 0.11)
    }
    
    # --- 4. EVOLUTION DYNAMIQUE ---
    wind_evol_params = {
        # Probabilité de changement : De 0% (Stable) à 90% (Chaos)
        'wind_change_prob': 1.0,
        'pattern_scale': 128,
        'perturbation_angle_amplitude': np.clip(0.085 + (0.15 * difficulty), 0, .12)*(difficulty > 0),
        'perturbation_strength_amplitude': np.clip(0.085 + (0.15 * difficulty), 0, .12) *(difficulty > 0),
        
        
        'rotation_bias': 0.01 + np.random.uniform(-0.045, 0.045) * difficulty,
        'bias_strength': np.clip(difficulty + 0.75, 0, 1.0) 
    }
    
    return wind_init_params, wind_evol_params

# =============================================================================
# PHYSICS FEATURES
# =============================================================================

def compute_physics_features(obs: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """Compute physics-based features from observation."""
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    
    MAX_DIST = 50.0
    MAX_SPEED = 7.0
    MAX_WIND = 7.0
    
    dx = goal[0] - x
    dy = goal[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > 0:
        dir_goal_x = dx / dist
        dir_goal_y = dy / dist
    else:
        dir_goal_x, dir_goal_y = 0, 0
        
    feat_dist = dist / MAX_DIST
    
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.001:
        dir_boat_x = vx / speed
        dir_boat_y = vy / speed
    else:
        dir_boat_x, dir_boat_y = 0, 0

    feat_speed = speed / MAX_SPEED
    
    wind_speed = np.sqrt(wx**2 + wy**2)
    if wind_speed > 0:
        dir_wind_x = wx / wind_speed
        dir_wind_y = wy / wind_speed
    else:
        dir_wind_x, dir_wind_y = 0, 0
        
    feat_wind_str = wind_speed / MAX_WIND
    
    feat_align_goal = (dir_boat_x * dir_goal_x) + (dir_boat_y * dir_goal_y)
    feat_wind_goal = (dir_wind_x * dir_goal_x) + (dir_wind_y * dir_goal_y)
    feat_angle_wind = (dir_boat_x * dir_wind_x) + (dir_boat_y * dir_wind_y)
    feat_cross_goal = (dir_boat_x * dir_goal_y) - (dir_boat_y * dir_goal_x)
    feat_cross_wind = (dir_boat_x * dir_wind_y) - (dir_boat_y * dir_wind_x)
    
    features = np.array([
        feat_dist, feat_speed, feat_wind_str,
        feat_align_goal, feat_wind_goal, feat_angle_wind,
        feat_cross_goal, feat_cross_wind,
        dir_boat_x, dir_boat_y, dir_wind_x, dir_wind_y
    ], dtype=np.float32)
    
    return features


# =============================================================================
# NOISY LINEAR
# =============================================================================

class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            return F.linear(x,
                           self.weight_mu + self.weight_sigma * self.weight_epsilon,
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


# =============================================================================
# WIND PATCH ENCODER (ViT-style)
# =============================================================================

class WindPatchEncoder(nn.Module):
    """
    Encode wind field into patch tokens (Vision Transformer style).
    
    32×32×2 → 64 patches (4×4 each) → 64 tokens of dim d_model
    """
    
    def __init__(self, patch_size=4, d_model=64, grid_size=32):
        super().__init__()
        
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.n_patches = (grid_size // patch_size) ** 2  # 64 patches
        self.n_patches_per_row = grid_size // patch_size  # 8
        
        # Patch projection
        patch_dim = 2 * patch_size * patch_size  # 32
        self.patch_proj = nn.Linear(patch_dim, d_model)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, wind_field):
        """
        Args:
            wind_field: (batch, 2, 32, 32)
        Returns:
            tokens: (batch, 64, d_model)
        """
        batch_size = wind_field.size(0)
        ps = self.patch_size
        
        # Extract patches using unfold
        # (B, 2, 32, 32) → (B, 2, 8, 4, 8, 4)
        x = wind_field.unfold(2, ps, ps).unfold(3, ps, ps)
        # (B, 2, 8, 8, 4, 4) → (B, 8, 8, 2, 4, 4) → (B, 64, 32)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, self.n_patches, -1)
        
        # Project to d_model
        tokens = self.patch_proj(x)
        
        # Add positional embedding
        tokens = tokens + self.pos_embedding
        
        # LayerNorm
        tokens = self.norm(tokens)
        
        return tokens


# =============================================================================
# STATE ENCODER (Creates Queries)
# =============================================================================

class StateEncoder(nn.Module):
    """
    Encode agent state (physics features) into query vectors.
    
    Creates multiple queries for different aspects:
    - Route query: Where to go?
    - Wind query: How is the wind on my path?
    - Efficiency query: Which direction is efficient?
    """
    
    def __init__(self, n_physics=12, d_model=64, n_queries=4):
        super().__init__()
        
        self.n_queries = n_queries
        self.d_model = d_model
        
        # Physics encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(n_physics, 64),
            nn.ReLU(),
            nn.Linear(64, d_model * n_queries)
        )
        
        # Learnable query templates
        self.learned_queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, physics):
        """
        Args:
            physics: (batch, n_physics)
        Returns:
            queries: (batch, n_queries, d_model)
        """
        batch_size = physics.size(0)
        
        # Encode physics → modulation
        physics_encoded = self.physics_encoder(physics)
        physics_encoded = physics_encoded.view(batch_size, self.n_queries, self.d_model)
        
        # Combine with learned queries
        queries = self.learned_queries + physics_encoded
        queries = self.norm(queries)
        
        return queries


# =============================================================================
# MULTI-HEAD CROSS-ATTENTION
# =============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention between queries (agent state) and wind tokens.
    """
    
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values):
        """
        Args:
            queries: (batch, n_queries, d_model)
            keys: (batch, n_patches, d_model)
            values: (batch, n_patches, d_model)
        Returns:
            output: (batch, n_queries, d_model)
        """
        batch_size = queries.size(0)
        n_q = queries.size(1)
        n_k = keys.size(1)
        
        # Project
        Q = self.q_proj(queries)
        K = self.k_proj(keys)
        V = self.v_proj(values)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply to values
        output = torch.matmul(attention, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, n_q, self.d_model)
        output = self.out_proj(output)
        
        return output


# =============================================================================
# TRANSFORMER LAYER
# =============================================================================

class TransformerLayer(nn.Module):
    """Transformer layer with cross-attention and FFN."""
    
    def __init__(self, d_model=64, n_heads=4, ffn_dim=256, dropout=0.1):
        super().__init__()
        
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, queries, wind_tokens):
        # Cross-attention with residual
        attn_out = self.cross_attn(queries, wind_tokens, wind_tokens)
        queries = self.norm1(queries + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries


# =============================================================================
# ATTENTION Q-NETWORK
# =============================================================================

class AttentionQNetwork(nn.Module):
    """
    Attention-based Q-Network (ViT-style).
    
    Architecture:
    1. Wind Patch Encoder: 32×32×2 → 64 tokens
    2. State Encoder: physics → n_queries queries
    3. Transformer Layers: cross-attention
    4. Dueling Head: Q = V + A - mean(A)
    """
    
    def __init__(self,
                 n_physics=12,
                 d_model=64,
                 n_heads=4,
                 n_layers=2,
                 n_queries=4,
                 patch_size=4,
                 use_noisy=True,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_queries = n_queries
        self.use_noisy = use_noisy
        
        # Encoders
        self.wind_encoder = WindPatchEncoder(patch_size=patch_size, d_model=d_model)
        self.state_encoder = StateEncoder(n_physics=n_physics, d_model=d_model, n_queries=n_queries)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Dueling head
        feature_dim = d_model * n_queries
        Linear = NoisyLinear if use_noisy else nn.Linear
        
        self.value_stream = nn.Sequential(
            Linear(feature_dim, 128),
            nn.ReLU(),
            Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            Linear(feature_dim, 128),
            nn.ReLU(),
            Linear(128, 9)
        )
    
    def forward(self, wind_field, physics):
        """
        Args:
            wind_field: (batch, 2, 32, 32)
            physics: (batch, n_physics)
        Returns:
            q_values: (batch, 9)
        """
        # Encode wind field → tokens
        wind_tokens = self.wind_encoder(wind_field)
        
        # Encode state → queries
        queries = self.state_encoder(physics)
        
        # Transformer layers
        for layer in self.layers:
            queries = layer(queries, wind_tokens)
        
        # Flatten queries
        features = queries.view(queries.size(0), -1)
        
        # Dueling head
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in NoisyLinear layers."""
        if not self.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# =============================================================================
# PRIORITIZED REPLAY BUFFER
# =============================================================================

class SumTree:
    """Sum tree for PER."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, device='cpu'):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 0.01
        self.abs_err_upper = 1.0
        self.device = device
    
    def push(self, wind_field, physics, action, reward, next_wind_field, next_physics, done):
        transition = (wind_field, physics, action, reward, next_wind_field, next_physics, done)
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, transition)
    
    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        weights /= weights.max()
        
        wind_list, physics_list, actions, rewards, next_wind_list, next_physics_list, dones = zip(*batch)
        
        wind = torch.FloatTensor(np.array(wind_list)).permute(0, 3, 1, 2).to(self.device)
        physics = torch.FloatTensor(np.array(physics_list)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_wind = torch.FloatTensor(np.array(next_wind_list)).permute(0, 3, 1, 2).to(self.device)
        next_physics = torch.FloatTensor(np.array(next_physics_list)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        return wind, physics, actions, rewards, next_wind, next_physics, dones, indices, weights_tensor
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# =============================================================================
# STANDARD REPLAY BUFFER (fallback)
# =============================================================================

class ReplayBuffer:
    """Standard replay buffer."""
    
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        self.wind_fields = np.zeros((capacity, 32, 32, 2), dtype=np.float32)
        self.physics = np.zeros((capacity, 12), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_wind_fields = np.zeros((capacity, 32, 32, 2), dtype=np.float32)
        self.next_physics = np.zeros((capacity, 12), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, wind_field, physics, action, reward, next_wind_field, next_physics, done):
        idx = self.position
        self.wind_fields[idx] = wind_field
        self.physics[idx] = physics
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_wind_fields[idx] = next_wind_field
        self.next_physics[idx] = next_physics
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        wind = torch.tensor(self.wind_fields[indices].transpose(0, 3, 1, 2), device=self.device)
        next_wind = torch.tensor(self.next_wind_fields[indices].transpose(0, 3, 1, 2), device=self.device)
        physics = torch.tensor(self.physics[indices], device=self.device)
        actions = torch.tensor(self.actions[indices], device=self.device)
        rewards = torch.tensor(self.rewards[indices], device=self.device)
        next_physics = torch.tensor(self.next_physics[indices], device=self.device)
        dones = torch.tensor(self.dones[indices], device=self.device)
        
        return wind, physics, actions, rewards, next_wind, next_physics, dones
    
    def __len__(self):
        return self.size


# =============================================================================
# AGENT (for inference/submission)
# =============================================================================

class MyAgentAttention(BaseAgent):
    """Attention-based DQN Agent for submission."""
    
    def __init__(self, model_path: Optional[str] = None, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.goal = (16, 31)
        
        self.q_network = AttentionQNetwork(
            n_physics=12,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_queries=4,
            patch_size=4,
            use_noisy=True
        ).to(self.device)
        
        if model_path:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.q_network.eval()
    
    def act(self, observation: np.ndarray) -> int:
        wind_field = observation[6:].reshape(32, 32, 2)
        physics = compute_physics_features(observation, self.goal)
        
        wind_tensor = torch.tensor(
            wind_field.transpose(2, 0, 1),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        physics_tensor = torch.tensor(
            physics, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def reset(self):
        pass
    
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)


# =============================================================================
# TRAINER
# =============================================================================

class DQNTrainer:
    """Trainer for Attention-based DQN."""
    
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        lr_decay=0.9999,
        gamma=0.99,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000,
        use_double_dqn=True,
        use_noisy_net=True,
        use_per=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100000,
        # Attention-specific
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_queries=4,
        patch_size=4,
        dropout=0.1,
        device='cpu',
        tensorboard_dir=None,
        train_scenarios=None,
        gradient_clip=10.0
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_noisy_net = use_noisy_net
        self.use_per = use_per
        self.gradient_clip = gradient_clip
        self.train_scenarios = train_scenarios or ['training_1']
        
        self.goal = (env.goal_position[0], env.goal_position[1])
        
        # Create networks
        self.q_network = AttentionQNetwork(
            n_physics=12,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_queries=n_queries,
            patch_size=patch_size,
            use_noisy=use_noisy_net,
            dropout=dropout
        ).to(device)
        
        self.target_network = AttentionQNetwork(
            n_physics=12,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_queries=n_queries,
            patch_size=patch_size,
            use_noisy=use_noisy_net,
            dropout=dropout
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        
        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_capacity, alpha=per_alpha,
                beta_start=per_beta_start, beta_frames=per_beta_frames,
                device=device
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_capacity, device)
        
        # Logging
        self.writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None
        
        # Counters
        self.steps = 0
        self.episodes = 0
        self.prev_distance = None
        
        print(f"🎮 Attention DQN Trainer initialized:")
        print(f"   d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")
        print(f"   n_queries: {n_queries}, patch_size: {patch_size}")
        print(f"   NoisyNet: {use_noisy_net}, PER: {use_per}, Double: {use_double_dqn}")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def compute_shaped_reward(self, obs, next_obs, raw_reward, done):
        """Reward shaping."""
        x, y = next_obs[0], next_obs[1]
        current_dist = np.sqrt((self.goal[0] - x)**2 + (self.goal[1] - y)**2)
        
        shaped_reward = raw_reward
        
        if self.prev_distance is not None:
            progress = self.prev_distance - current_dist
            shaped_reward += progress * 10
        
        self.prev_distance = current_dist
        
        vx, vy = next_obs[2], next_obs[3]
        speed = np.sqrt(vx**2 + vy**2)
        shaped_reward += speed * 0.9 - 1
        
        if done:
            self.prev_distance = None
        
        return shaped_reward
    
    def get_action(self, obs, greedy=False):
        """Select action."""
        wind_field = obs[6:].reshape(32, 32, 2)
        wind_tensor = torch.FloatTensor(wind_field).permute(2, 0, 1).to(self.device).unsqueeze(0)
        
        physics = compute_physics_features(obs, self.goal)
        physics_tensor = torch.FloatTensor(physics).to(self.device).unsqueeze(0)
        
        if greedy:
            self.q_network.eval()
        else:
            self.q_network.train()
        
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def train_step(self):
        """One training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample
        if self.use_per:
            wind, physics, actions, rewards, next_wind, next_physics, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
        else:
            wind, physics, actions, rewards, next_wind, next_physics, dones = \
                self.replay_buffer.sample(self.batch_size)
        
        # Q(s, a)
        q_values = self.q_network(wind, physics)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_wind, next_physics).argmax(1)
                next_q_values = self.target_network(next_wind, next_physics)
                next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_wind, next_physics)
                next_q_value = next_q_values.max(1).values
            
            target = rewards + self.gamma * next_q_value * (1 - dones)
        
        # Loss
        if self.use_per:
            td_errors = (q_value - target).detach().cpu().numpy()
            loss = (weights * F.smooth_l1_loss(q_value, target, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(q_value, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update PER priorities
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Reset noise
        if self.use_noisy_net:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Log
        if self.writer and self.steps % 100 == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), self.steps)
            with torch.no_grad():
                self.writer.add_scalar('Train/MeanQ', q_values.mean().item(), self.steps)
                self.writer.add_scalar('Train/MaxQ', q_values.max().item(), self.steps)
                # Q-value spread (important metric!)
                q_spread = (q_values.max(1).values - q_values.min(1).values).mean().item()
                self.writer.add_scalar('Train/QSpread', q_spread, self.steps)
        
        # Update target
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"   Target network updated at step {self.steps}")
        
        return loss.item()
    
    def collect_episode(self, max_steps=200):
        """Collect one episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        self.prev_distance = None
        
        for step in range(max_steps):
            action = self.get_action(obs)
            next_obs, raw_reward, done, truncated, _ = self.env.step(action)
            
            shaped_reward = self.compute_shaped_reward(obs, next_obs, raw_reward, done or truncated)
            
            wind_field = obs[6:].reshape(32, 32, 2)
            next_wind_field = next_obs[6:].reshape(32, 32, 2)
            physics = compute_physics_features(obs, self.goal)
            next_physics = compute_physics_features(next_obs, self.goal)
            
            self.replay_buffer.push(
                wind_field, physics, action, shaped_reward,
                next_wind_field, next_physics, done or truncated
            )
            
            episode_reward += shaped_reward
            obs = next_obs
            self.steps += 1
            
            if self.steps % 4 == 0:
                self.train_step()
            
            if done or truncated:
                break
        
        self.episodes += 1
        return episode_reward
    
    def evaluate(self, n_episodes=5, max_steps=200):
        """Evaluate agent."""
        total_reward = 0
        successes = 0
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(max_steps):
                action = self.get_action(obs, greedy=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    if reward > 0:
                        successes += 1
                    break
                if truncated:
                    break
            
            total_reward += episode_reward
        
        return total_reward / n_episodes, successes / n_episodes
    
    def train(self, num_episodes, eval_freq=100, save_freq=500, verbose=True):
        """Main training loop."""
        best_eval_reward = -np.inf
        
        for episode in range(num_episodes):
            # Curriculum
            progress = episode / num_episodes
            init_params, evol_params = generate_curriculum_params(progress)
            self.env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
            
            # Collect
            episode_reward = self.collect_episode()
            
            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], episode)
                self.writer.add_scalar('Train/BufferSize', len(self.replay_buffer), episode)
            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Steps: {self.steps}")
            
            # Eval
            if episode % eval_freq == 0 and episode > 0:
                eval_reward, success_rate = self.evaluate(n_episodes=10)
                
                if self.writer:
                    self.writer.add_scalar('Eval/Reward', eval_reward, episode)
                    self.writer.add_scalar('Eval/SuccessRate', success_rate, episode)
                
                print(f"[EVAL] Episode {episode} | Reward: {eval_reward:.2f} | Success: {success_rate:.1%}")
            
            # Save
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'checkpoints/attention/checkpoint_ep{episode}.pth')
                print(f"💾 Checkpoint saved: episode {episode}")
        
        final_path = 'checkpoints/attention/final_model.pth'
        self.save_model(final_path)
        print(f"💾 Final model saved: {final_path}")
        
        if self.writer:
            self.writer.close()
    
    def save_model(self, path='attention_model.pth'):
        """Save model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network.state_dict(), path)