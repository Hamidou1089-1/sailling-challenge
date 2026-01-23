"""
Utility to save Attention DQN agents for Codabench submission (Pure NumPy).

Converts the PyTorch Attention network to pure NumPy implementation.
"""

import os
import numpy as np
import torch


def array_to_string(arr, name="array"):
    """Convert numpy array to string representation."""
    if arr.ndim == 1:
        return np.array2string(arr, separator=', ', max_line_width=100, 
                              threshold=np.inf, precision=6)
    else:
        return np.array2string(arr, separator=', ', max_line_width=100,
                              threshold=np.inf, precision=6)


def save_attention_agent(trainer, output_path, agent_class_name="MyAgent"):
    """
    Save an Attention-based DQN agent for Codabench submission (Pure NumPy).
    
    Args:
        trainer: The trained DQNTrainer instance
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING ATTENTION AGENT FOR CODABENCH (PURE NUMPY)")
    print(f"{'='*70}")
    
    model = trainer.q_network
    
    # Get config
    d_model = model.d_model
    n_queries = model.n_queries
    n_heads = 4  # Default
    n_layers = len(model.layers)
    
    print(f"\n🔍 Architecture:")
    print(f"   d_model: {d_model}")
    print(f"   n_queries: {n_queries}")
    print(f"   n_heads: {n_heads}")
    print(f"   n_layers: {n_layers}")
    
    # Extract weights (skip noise parameters)
    weights_dict = {}
    print("\n📦 Extracting weights...")
    
    for name, param in model.state_dict().items():
        if 'epsilon' in name or 'sigma' in name:
            continue
        
        clean_name = name.replace('_mu', '')
        weights_dict[clean_name] = param.cpu().detach().numpy()
        print(f"   ✓ {clean_name}: {weights_dict[clean_name].shape}")
    
    goal = trainer.goal
    total_params = sum(w.size for w in weights_dict.values())
    
    print(f"\n📊 Configuration:")
    print(f"   Goal: {goal}")
    print(f"   Total parameters: {total_params:,}")
    
    # Generate code
    file_content = generate_numpy_code(weights_dict, goal, d_model, n_queries, n_heads, n_layers)
    
    print(f"\n💾 Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print("✅ AGENT SAVED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"📄 Output: {output_path}")
    print(f"📊 Size: {file_size_mb:.2f} MB")
    print(f"🧠 Parameters: {total_params:,}")
    print(f"\n📋 Next steps:")
    print(f"   1. Test: python {output_path}")
    print(f"   2. Validate: python src/test_agent_validity.py {output_path}")
    print(f"   3. Submit to Codabench")


def generate_numpy_code(weights_dict, goal, d_model, n_queries, n_heads, n_layers):
    """Generate the NumPy implementation code."""
    
    head_dim = d_model // n_heads
    
    code = f'''"""
Attention-Based DQN Agent for Sailing Challenge (Pure NumPy)

Architecture (ViT-style):
- Wind Patches: 32×32×2 → 64 tokens (4×4 patches)
- State Encoder: 12 physics → {n_queries} queries
- Cross-Attention: {n_layers} layers, {n_heads} heads
- Dueling Head: Q = V + A - mean(A)

Total parameters: {sum(w.size for w in weights_dict.values()):,}
"""

import numpy as np
from typing import Tuple

from evaluator.base_agent import BaseAgent


# =============================================================================
# PHYSICS FEATURES
# =============================================================================

def compute_physics_features(obs: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """Compute physics features from observation."""
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    
    MAX_DIST, MAX_SPEED, MAX_WIND = 50.0, 7.0, 7.0
    
    dx, dy = goal[0] - x, goal[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > 0:
        dir_goal_x, dir_goal_y = dx / dist, dy / dist
    else:
        dir_goal_x, dir_goal_y = 0.0, 0.0
    
    feat_dist = dist / MAX_DIST
    
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.001:
        dir_boat_x, dir_boat_y = vx / speed, vy / speed
    else:
        dir_boat_x, dir_boat_y = 0.0, 0.0
    feat_speed = speed / MAX_SPEED
    
    wind_speed = np.sqrt(wx**2 + wy**2)
    if wind_speed > 0:
        dir_wind_x, dir_wind_y = wx / wind_speed, wy / wind_speed
    else:
        dir_wind_x, dir_wind_y = 0.0, 0.0
    feat_wind_str = wind_speed / MAX_WIND
    
    feat_align_goal = dir_boat_x * dir_goal_x + dir_boat_y * dir_goal_y
    feat_wind_goal = dir_wind_x * dir_goal_x + dir_wind_y * dir_goal_y
    feat_angle_wind = dir_boat_x * dir_wind_x + dir_boat_y * dir_wind_y
    feat_cross_goal = dir_boat_x * dir_goal_y - dir_boat_y * dir_goal_x
    feat_cross_wind = dir_boat_x * dir_wind_y - dir_boat_y * dir_wind_x
    
    return np.array([
        feat_dist, feat_speed, feat_wind_str,
        feat_align_goal, feat_wind_goal, feat_angle_wind,
        feat_cross_goal, feat_cross_wind,
        dir_boat_x, dir_boat_y, dir_wind_x, dir_wind_y
    ], dtype=np.float32)


# =============================================================================
# NUMPY OPERATIONS
# =============================================================================

def softmax(x, axis=-1):
    """Softmax along axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


# =============================================================================
# WIND ENCODER
# =============================================================================

class NumpyWindEncoder:
    """Wind patch encoder in NumPy."""
    
    def __init__(self, weights):
        self.w = weights
        self.patch_size = 4
        self.n_patches = 64
    
    def forward(self, wind_field):
        """
        Args:
            wind_field: (batch, 2, 32, 32)
        Returns:
            tokens: (batch, 64, d_model)
        """
        batch_size = wind_field.shape[0]
        ps = self.patch_size
        
        # Extract patches: (batch, 64, 32)
        patches = []
        for i in range(8):
            for j in range(8):
                patch = wind_field[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
                patches.append(patch.reshape(batch_size, -1))
        patches = np.stack(patches, axis=1)
        
        # Project
        tokens = np.dot(patches, self.w['wind_encoder.patch_proj.weight'].T)
        tokens = tokens + self.w['wind_encoder.patch_proj.bias']
        
        # Add positional embedding
        tokens = tokens + self.w['wind_encoder.pos_embedding']
        
        # LayerNorm
        tokens = layer_norm(tokens, 
                           self.w['wind_encoder.norm.weight'],
                           self.w['wind_encoder.norm.bias'])
        
        return tokens


# =============================================================================
# STATE ENCODER
# =============================================================================

class NumpyStateEncoder:
    """State encoder in NumPy."""
    
    def __init__(self, weights, n_queries={n_queries}, d_model={d_model}):
        self.w = weights
        self.n_queries = n_queries
        self.d_model = d_model
    
    def forward(self, physics):
        """
        Args:
            physics: (batch, 12)
        Returns:
            queries: (batch, n_queries, d_model)
        """
        batch_size = physics.shape[0]
        
        # MLP
        x = np.dot(physics, self.w['state_encoder.physics_encoder.0.weight'].T)
        x = x + self.w['state_encoder.physics_encoder.0.bias']
        x = relu(x)
        x = np.dot(x, self.w['state_encoder.physics_encoder.2.weight'].T)
        x = x + self.w['state_encoder.physics_encoder.2.bias']
        
        # Reshape
        x = x.reshape(batch_size, self.n_queries, self.d_model)
        
        # Add learned queries
        queries = x + self.w['state_encoder.learned_queries']
        
        # LayerNorm
        queries = layer_norm(queries,
                            self.w['state_encoder.norm.weight'],
                            self.w['state_encoder.norm.bias'])
        
        return queries


# =============================================================================
# CROSS-ATTENTION
# =============================================================================

class NumpyCrossAttention:
    """Cross-attention in NumPy."""
    
    def __init__(self, weights, d_model={d_model}, n_heads={n_heads}):
        self.w = weights
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, queries, keys, values, prefix):
        """
        Cross-attention.
        
        Args:
            queries: (batch, n_q, d_model)
            keys: (batch, n_k, d_model)
            values: (batch, n_k, d_model)
            prefix: weight prefix
        
        Returns:
            (batch, n_q, d_model)
        """
        batch_size = queries.shape[0]
        n_q = queries.shape[1]
        n_k = keys.shape[1]
        
        # Project
        Q = np.dot(queries, self.w[f'{{prefix}}.q_proj.weight'].T) + self.w[f'{{prefix}}.q_proj.bias']
        K = np.dot(keys, self.w[f'{{prefix}}.k_proj.weight'].T) + self.w[f'{{prefix}}.k_proj.bias']
        V = np.dot(values, self.w[f'{{prefix}}.v_proj.weight'].T) + self.w[f'{{prefix}}.v_proj.bias']
        
        # Reshape for multi-head: (batch, n, heads, head_dim) -> (batch, heads, n, head_dim)
        Q = Q.reshape(batch_size, n_q, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, n_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, n_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale
        attention = softmax(scores, axis=-1)
        
        # Apply to values
        output = np.matmul(attention, V)
        
        # Reshape back: (batch, heads, n_q, head_dim) -> (batch, n_q, d_model)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, n_q, self.d_model)
        
        # Output projection
        output = np.dot(output, self.w[f'{{prefix}}.out_proj.weight'].T) + self.w[f'{{prefix}}.out_proj.bias']
        
        return output


# =============================================================================
# WEIGHTS
# =============================================================================

def load_weights():
    """Load weights from code."""
    w = {{}}
'''
    
    # Add weights
    for name, arr in weights_dict.items():
        arr_str = array_to_string(arr, name)
        code += f"    w['{name}'] = np.array({arr_str}, dtype=np.float32)"
        if arr.ndim > 1:
            code += f".reshape{arr.shape}"
        code += "\n"
    
    code += "    return w\n\n"
    
    # Agent class
    code += f'''
# =============================================================================
# AGENT
# =============================================================================

class MyAgent(BaseAgent):
    """Attention-based DQN Agent."""
    
    def __init__(self):
        super().__init__()
        self.goal = {goal}
        self.d_model = {d_model}
        self.n_queries = {n_queries}
        self.n_heads = {n_heads}
        self.n_layers = {n_layers}
        
        # Load weights
        self.w = load_weights()
        
        # Initialize components
        self.wind_encoder = NumpyWindEncoder(self.w)
        self.state_encoder = NumpyStateEncoder(self.w)
        self.attention = NumpyCrossAttention(self.w)
    
    def act(self, observation):
        """Select action with highest Q-value."""
        # Prepare inputs
        wind_field = observation[6:].reshape(32, 32, 2)
        wind_tensor = wind_field.transpose(2, 0, 1).astype(np.float32)
        wind_tensor = wind_tensor.reshape(1, 2, 32, 32)
        
        physics = compute_physics_features(observation, self.goal)
        physics_tensor = physics.reshape(1, -1)
        
        # Encode
        wind_tokens = self.wind_encoder.forward(wind_tensor)  # (1, 64, d)
        queries = self.state_encoder.forward(physics_tensor)  # (1, n_q, d)
        
        # Transformer layers
        for layer_idx in range(self.n_layers):
            # Cross-attention
            prefix = f'layers.{{layer_idx}}.cross_attn'
            attn_out = self.attention.forward(queries, wind_tokens, wind_tokens, prefix)
            
            # Residual + LayerNorm
            queries = queries + attn_out
            queries = layer_norm(queries,
                                self.w[f'layers.{{layer_idx}}.norm1.weight'],
                                self.w[f'layers.{{layer_idx}}.norm1.bias'])
            
            # FFN
            ffn_prefix = f'layers.{{layer_idx}}.ffn'
            ffn = np.dot(queries, self.w[f'{{ffn_prefix}}.0.weight'].T) + self.w[f'{{ffn_prefix}}.0.bias']
            ffn = relu(ffn)
            ffn = np.dot(ffn, self.w[f'{{ffn_prefix}}.3.weight'].T) + self.w[f'{{ffn_prefix}}.3.bias']
            
            # Residual + LayerNorm
            queries = queries + ffn
            queries = layer_norm(queries,
                                self.w[f'layers.{{layer_idx}}.norm2.weight'],
                                self.w[f'layers.{{layer_idx}}.norm2.bias'])
        
        # Flatten queries
        features = queries.reshape(1, -1)
        
        # Dueling head
        # Value stream
        v = np.dot(features, self.w['value_stream.0.weight'].T) + self.w['value_stream.0.bias']
        v = relu(v)
        v = np.dot(v, self.w['value_stream.2.weight'].T) + self.w['value_stream.2.bias']
        
        # Advantage stream
        a = np.dot(features, self.w['advantage_stream.0.weight'].T) + self.w['advantage_stream.0.bias']
        a = relu(a)
        a = np.dot(a, self.w['advantage_stream.2.weight'].T) + self.w['advantage_stream.2.bias']
        
        # Q = V + A - mean(A)
        q_values = v + a - np.mean(a, axis=-1, keepdims=True)
        
        return int(np.argmax(q_values[0]))
    
    def reset(self):
        """Reset agent for new episode."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
'''
    
    return code