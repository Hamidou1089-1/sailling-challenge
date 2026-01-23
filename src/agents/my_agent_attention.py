"""
Attention-Based DQN Agent for Sailing Challenge (ViT-style) - NO REPLAY BUFFER

Architecture:
- Wind Field → Patches → Tokens (comme Vision Transformer)
- Physics → Queries
- Cross-Attention: Queries attend aux Wind Tokens
- Dueling Head: Q = V + A - mean(A)

L'agent "regarde" les zones pertinentes du wind field selon son état.

TRAINING STRATEGY:
- NO REPLAY BUFFER: Chaque épisode est unique (curriculum génère différents environnements)
- Train on full episode: Collecte toutes les transitions, puis entraîne 3-4 fois dessus
- Chaque configuration de vent est précieuse et capturée directement
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
    theta = np.random.uniform(0, 2 * np.pi)
    wind_dir = (np.cos(theta), np.sin(theta))
    
    # --- 2. GESTION DE LA DIFFICULTÉ (Le "Recall") ---
    if np.random.random() < 0.3:
        difficulty = 0.04
    else:
        difficulty = 0.5 + np.random.uniform(-0.4, 0.45)

    # --- 3. PARAMÈTRES DU VENT ---
    base_speed = 3.0 + np.random.uniform(-0.01, 0.01)
    
    wind_init_params = {
        'base_speed': base_speed,
        'base_direction': wind_dir,
        'pattern_scale': np.random.choice([32, 32, 64, 128, 128]), 
        'pattern_strength': np.clip(0.08 + (0.75 * difficulty), 0.1, 0.62),
        'strength_variation': np.clip(0.08 + (0.5 * difficulty), 0.1, 0.42),
        'noise': np.clip(0.08 + (0.15 * difficulty), 0, 0.11)
    }
    
    # --- 4. EVOLUTION DYNAMIQUE ---
    wind_evol_params = {
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
    """Encode wind field into patch tokens (Vision Transformer style)."""
    
    def __init__(self, patch_size=4, d_model=64):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 32x32 grid with patches of size patch_size x patch_size
        self.n_patches = (32 // patch_size) ** 2
        patch_dim = 2 * patch_size * patch_size  # 2 channels (wx, wy)
        
        # Linear projection of flattened patches
        self.patch_proj = nn.Linear(patch_dim, d_model)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, wind_field):
        """
        Args:
            wind_field: (batch, 2, 32, 32) - wind vectors
        Returns:
            tokens: (batch, n_patches, d_model)
        """
        batch_size = wind_field.shape[0]
        ps = self.patch_size
        
        # Extract patches: (batch, n_patches, patch_dim)
        patches = wind_field.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, self.n_patches, -1)
        
        # Project to d_model
        tokens = self.patch_proj(patches)
        
        # Add positional embedding
        tokens = tokens + self.pos_embedding
        
        tokens = self.norm(tokens)
        
        return tokens


# =============================================================================
# STATE ENCODER (Physics → Queries)
# =============================================================================

class StateEncoder(nn.Module):
    """Encode physics state into query vectors."""
    
    def __init__(self, n_physics=12, n_queries=4, d_model=64):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        
        # Physics encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(n_physics, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_queries * d_model)
        )
        
        # Learnable query embeddings (like "aspects of decision")
        self.learned_queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, physics):
        """
        Args:
            physics: (batch, n_physics)
        Returns:
            queries: (batch, n_queries, d_model)
        """
        batch_size = physics.shape[0]
        
        # Encode physics
        encoded = self.physics_encoder(physics)
        encoded = encoded.view(batch_size, self.n_queries, self.d_model)
        
        # Add learned queries
        queries = encoded + self.learned_queries
        
        queries = self.norm(queries)
        
        return queries


# =============================================================================
# CROSS-ATTENTION FUSION
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention between state queries and wind tokens."""
    
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # Q from state, K/V from wind
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, queries, keys, values):
        """
        Cross-attention.
        
        Args:
            queries: (batch, n_q, d_model) - from state
            keys: (batch, n_k, d_model) - from wind
            values: (batch, n_k, d_model) - from wind
        
        Returns:
            (batch, n_q, d_model)
        """
        batch_size = queries.shape[0]
        n_q = queries.shape[1]
        n_k = keys.shape[1]
        
        # Project
        Q = self.q_proj(queries)
        K = self.k_proj(keys)
        V = self.v_proj(values)
        
        # Reshape for multi-head: (batch, n, d) -> (batch, n, heads, head_dim) -> (batch, heads, n, head_dim)
        Q = Q.view(batch_size, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply to values
        output = torch.matmul(attention, V)
        
        # Reshape back: (batch, heads, n_q, head_dim) -> (batch, n_q, heads, head_dim) -> (batch, n_q, d)
        output = output.transpose(1, 2).contiguous().view(batch_size, n_q, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


# =============================================================================
# TRANSFORMER LAYER
# =============================================================================

class TransformerLayer(nn.Module):
    """Transformer layer with cross-attention."""
    
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        
        self.cross_attn = CrossAttentionFusion(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, queries, wind_tokens):
        """
        Args:
            queries: (batch, n_q, d_model)
            wind_tokens: (batch, n_patches, d_model)
        Returns:
            (batch, n_q, d_model)
        """
        # Cross-attention
        attn_out = self.cross_attn(queries, wind_tokens, wind_tokens)
        queries = self.norm1(queries + attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries


# =============================================================================
# ATTENTION Q-NETWORK
# =============================================================================

class AttentionQNetwork(nn.Module):
    """Complete Attention-based Q-Network."""
    
    def __init__(
        self,
        n_physics=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_queries=4,
        patch_size=4,
        n_actions=9,
        use_noisy=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_queries = n_queries
        
        # Encoders
        self.wind_encoder = WindPatchEncoder(patch_size, d_model)
        self.state_encoder = StateEncoder(n_physics, n_queries, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Dueling head
        hidden_dim = n_queries * d_model
        
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        self.value_stream = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            LinearLayer(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            LinearLayer(hidden_dim // 2, n_actions)
        )
    
    def forward(self, wind_field, physics):
        """
        Args:
            wind_field: (batch, 2, 32, 32)
            physics: (batch, 12)
        Returns:
            q_values: (batch, 9)
        """
        # Encode
        wind_tokens = self.wind_encoder(wind_field)  # (batch, n_patches, d_model)
        queries = self.state_encoder(physics)  # (batch, n_queries, d_model)
        
        # Transformer layers
        for layer in self.layers:
            queries = layer(queries, wind_tokens)
        
        # Flatten queries
        features = queries.flatten(1)  # (batch, n_queries * d_model)
        
        # Dueling head
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


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
# TRAINER (NO REPLAY BUFFER)
# =============================================================================

class DQNTrainer:
    """Trainer for Attention-based DQN - PURE ONLINE LEARNING."""
    
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        lr_decay=0.9999,
        gamma=0.99,
        target_update_freq=1000,
        use_double_dqn=True,
        use_noisy_net=True,
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
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_noisy_net = use_noisy_net
        self.gradient_clip = gradient_clip
        self.train_scenarios = train_scenarios or ['training_1']
        
        self.goal = (env.goal_position[0], env.goal_position[1])
        
        # Create networks
        self.q_network = AttentionQNetwork(
            n_physics=12, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, n_queries=n_queries, patch_size=patch_size,
            use_noisy=use_noisy_net, dropout=dropout
        ).to(device)
        
        self.target_network = AttentionQNetwork(
            n_physics=12, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, n_queries=n_queries, patch_size=patch_size,
            use_noisy=use_noisy_net, dropout=dropout
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        
        # Logging
        self.writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None
        
        # Counters
        self.steps = 0
        self.episodes = 0
        self.prev_distance = None
        
        print(f"🎮 Attention DQN - PURE ONLINE LEARNING:")
        print(f"   Train on EVERY transition immediately")
        print(f"   No buffer, no batches, no epochs")
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
    
    def train_step(self, wind, physics, action, reward, next_wind, next_physics, done):
        """Train on a SINGLE transition."""
        # Convert to tensors (batch_size = 1)
        wind_t = torch.FloatTensor(wind).permute(2, 0, 1).to(self.device).unsqueeze(0)
        physics_t = torch.FloatTensor(physics).to(self.device).unsqueeze(0)
        action_t = torch.LongTensor([action]).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)
        next_wind_t = torch.FloatTensor(next_wind).permute(2, 0, 1).to(self.device).unsqueeze(0)
        next_physics_t = torch.FloatTensor(next_physics).to(self.device).unsqueeze(0)
        done_t = torch.FloatTensor([done]).to(self.device)
        
        # Q(s, a)
        q_values = self.q_network(wind_t, physics_t)
        q_value = q_values[0, action_t[0]]
        
        # Target
        with torch.no_grad():
            if self.use_double_dqn:
                next_action = self.q_network(next_wind_t, next_physics_t).argmax(1)[0]
                next_q_values = self.target_network(next_wind_t, next_physics_t)
                next_q_value = next_q_values[0, next_action]
            else:
                next_q_values = self.target_network(next_wind_t, next_physics_t)
                next_q_value = next_q_values.max(1).values[0]
            
            target = reward_t[0] + self.gamma * next_q_value * (1 - done_t[0])
        
        # Loss
        loss = F.smooth_l1_loss(q_value, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        # Reset noise
        if self.use_noisy_net:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            if self.steps > 0:
                print(f"   Target network updated at step {self.steps}")
        
        # Log
        if self.writer and self.steps % 100 == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), self.steps)
            self.writer.add_scalar('Train/MeanQ', q_values.mean().item(), self.steps)
            self.writer.add_scalar('Train/MaxQ', q_values.max().item(), self.steps)
        
        return loss.item()
    
    def collect_episode(self, max_steps=200):
        """Collect one episode and train on EACH transition."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        self.prev_distance = None
        n_steps = 0
        
        for step in range(max_steps):
            action = self.get_action(obs)
            next_obs, raw_reward, done, truncated, _ = self.env.step(action)
            
            shaped_reward = self.compute_shaped_reward(obs, next_obs, raw_reward, done or truncated)
            
            # Extract features
            wind_field = obs[6:].reshape(32, 32, 2)
            next_wind_field = next_obs[6:].reshape(32, 32, 2)
            physics = compute_physics_features(obs, self.goal)
            next_physics = compute_physics_features(next_obs, self.goal)
            
            # Train immediately on this transition
            loss = self.train_step(
                wind_field, physics, action, shaped_reward,
                next_wind_field, next_physics, done or truncated
            )
            
            episode_reward += shaped_reward
            episode_loss += loss
            n_steps += 1
            self.steps += 1
            
            obs = next_obs
            
            if done or truncated:
                break
        
        # Update learning rate once per episode
        self.scheduler.step()
        
        avg_loss = episode_loss / n_steps if n_steps > 0 else 0
        
        if self.writer:
            self.writer.add_scalar('Train/EpisodeReward', episode_reward, self.episodes)
            self.writer.add_scalar('Train/EpisodeLoss', avg_loss, self.episodes)
            self.writer.add_scalar('Train/EpisodeLength', n_steps, self.episodes)
            self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.episodes)
        
        self.episodes += 1
        return episode_reward
    
    def evaluate(self, n_episodes=5, max_steps=200, eval_scenario='training_1'):
        """Evaluate agent on a standard scenario."""
        total_reward = 0
        successes = 0
        
        from wind_scenarios import get_wind_scenario
        eval_env = SailingEnv(**get_wind_scenario(eval_scenario))
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            
            for _ in range(max_steps):
                action = self.get_action(obs, greedy=True)
                obs, reward, done, truncated, info = eval_env.step(action)
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
        for episode in range(num_episodes):
            # Curriculum
            progress = episode / num_episodes
            init_params, evol_params = generate_curriculum_params(progress)
            self.env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
            
            # Collect and train
            episode_reward = self.collect_episode()
            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6e} | "
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
