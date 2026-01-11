import numpy as np
import sys
import os

import wind_scenarios

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_agent import BaseAgent
from typing import Dict, Any, Tuple, Optional

import torch.nn as nn
import torch
from torch.optim import AdamW
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
from wind_scenarios.env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario

from torch.optim.lr_scheduler import ExponentialLR

import pickle
from wind_scenarios.sailing_physics import calculate_sailing_efficiency 
import math



# def generate_curriculum_params(progress):
#     """
#     Générateur d'environnement robuste pour généralisation.
    
#     Args:
#         progress (float): 0.0 (Début) -> 1.0 (Fin)
#     """
    
#     # --- 1. DIRECTION : Toujours Aléatoire 360° ---
#     # C'est CRUCIAL. L'agent doit comprendre que le vent peut venir de n'importe où,
#     # même à l'épisode 1. La facilité vient de la stabilité, pas de la direction.
#     theta = np.random.uniform(0, 2 * np.pi)
#     wind_dir = (np.cos(theta), np.sin(theta))
    
#     # --- 2. GESTION DE LA DIFFICULTÉ (Le "Recall") ---
#     # On garde 20-30% d'épisodes "Faciles" (Vent stable) tout le temps.
#     # Cela sert d'ancrage pour que l'agent n'oublie pas les bases.
#     if np.random.random() < 0.25:
#         difficulty = 0.1  # Mode "Repos / Fondamentaux"
#     else:
#         # La difficulté suit la progression. 
#         # On ajoute un petit bruit pour ne pas être trop linéaire.
#         difficulty = np.clip(progress + np.random.uniform(-0.2, 0.5), 0.0, 1.0)

#     # --- 3. PARAMÈTRES DU VENT ---
    
#     # Vitesse : 3.0 est la vitesse standard. 
#     # Plus c'est dur, plus on s'éloigne de cette norme (vent très faible ou tempête).
#     # difficulty 0 -> speed 3.0
#     # difficulty 1 -> speed entre 1.0 et 5.0
#     speed_noise = np.random.uniform(-2.0, 2.0) * difficulty
#     base_speed = 3.0 + speed_noise
    
#     wind_init_params = {
#         'base_speed': base_speed,
#         'base_direction': wind_dir,
        
#         # Echelle : 128 (Large/Facile) -> 16 (Haché/Dur)
#         'pattern_scale': 128 - int(122 * difficulty), 
        
#         # Force des turbulences
#         'pattern_strength': 0.2 + (0.6 * difficulty),
#         'strength_variation': 0.2 + (0.6 * difficulty),
#         'noise': 0.03 + (0.25 * difficulty)
#     }
    
#     # --- 4. EVOLUTION DYNAMIQUE ---
#     wind_evol_params = {
#         # Probabilité de changement : De 0% (Stable) à 90% (Chaos)
#         'wind_change_prob': np.clip(0.3 + (0.75 * difficulty), 0.3, 1.0),
#         'pattern_scale': 64,
#         'perturbation_angle_amplitude': 0.08 + (0.15 * difficulty),
#         'perturbation_strength_amplitude': 0.08 + (0.15 * difficulty),
        
#         # Rotation du vent (Le tueur d'agent)
#         # difficulty 0 -> rotation 0
#         # difficulty 1 -> rotation +/- 0.025 rad par step
#         'rotation_bias': 0.01 + np.random.uniform(-0.08, 0.08) * (0.35 + difficulty),
#         'bias_strength': difficulty*1.5
#     }
    
#     return wind_init_params, wind_evol_params


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
    
    # --- 2. GESTION DE LA DIFFICULTÉ (Le "Recall") ---
    # On garde 20-30% d'épisodes "Faciles" (Vent stable) tout le temps.
    # Cela sert d'ancrage pour que l'agent n'oublie pas les bases.
    if np.random.random() < 0.2:
        difficulty = 0.0  # Mode "Repos / Fondamentaux"
    else:
        # La difficulté suit la progression. 
        # On ajoute un petit bruit pour ne pas être trop linéaire.
        difficulty = np.clip(progress + np.random.uniform(-1, 1), 0.0, 1.0)

    # --- 3. PARAMÈTRES DU VENT ---
    
    # Vitesse : 3.0 est la vitesse standard. 
    # Plus c'est dur, plus on s'éloigne de cette norme (vent très faible ou tempête).
    # difficulty 0 -> speed 3.0
    # difficulty 1 -> speed entre 1.0 et 5.0
    speed_noise = np.random.uniform(-2.0, 2.0) * difficulty
    base_speed = 3.0 + speed_noise
    
    wind_init_params = {
        'base_speed': base_speed,
        'base_direction': wind_dir,
        
        # Echelle : 128 (Large/Facile) -> 16 (Haché/Dur)
        'pattern_scale': np.clip(128 - int(122 * difficulty), 32, 128), 
        
        # Force des turbulences
        'pattern_strength': 0.2 + (0.5 * difficulty),
        'strength_variation': 0.15 + (0.5 * difficulty),
        'noise': 0.085 + (0.05 * difficulty)
    }
    
    # --- 4. EVOLUTION DYNAMIQUE ---
    wind_evol_params = {
        # Probabilité de changement : De 0% (Stable) à 90% (Chaos)
        'wind_change_prob': np.clip(0.15 + (0.75 * difficulty), 0, 1) * (difficulty > 0),
        'pattern_scale': 128,
        'perturbation_angle_amplitude': np.clip(0.085 + (0.15 * difficulty), 0, 1)*(difficulty > 0),
        'perturbation_strength_amplitude': np.clip(0.085 + (0.15 * difficulty), 0, 1) *(difficulty > 0),
        
        
        'rotation_bias': np.random.uniform(-0.045, 0.045) * difficulty,
        'bias_strength': np.clip(difficulty + 0.15, 0, 1.0) * (difficulty > 0)
    }
    
    return wind_init_params, wind_evol_params



def compute_physics_features(obs: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """
    Calcule les physics features normalisées avec mean/std.
    
    Args:
        obs: Observation brute
        goal: Position du goal
        feature_mean: Mean des features (pré-calculé)
        feature_std: Std des features (pré-calculé)
    
    Returns:
        Physics features normalisées (15 features)
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5] # Vent local
    
    # Constantes physiques (Hardcodées larges pour être sûr)
    MAX_DIST = 50.0  # Diagonale de la map
    MAX_SPEED = 6.0  # Vitesse max raisonnable du bateau
    MAX_WIND = 6.0   # Vitesse max du vent
    
    # --- A. POSITION RELATIVE (Vecteur vers le but) ---
    dx = goal[0] - x
    dy = goal[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    
    # Vecteur direction vers le but (Normalisé)
    if dist > 0:
        dir_goal_x = dx / dist
        dir_goal_y = dy / dist
    else:
        dir_goal_x, dir_goal_y = 0, 0
        
    # Feature 1: Distance normalisée (0 à 1)
    feat_dist = np.clip(dist / MAX_DIST, 0, 1)
    
    # --- B. VITESSE BATEAU ---
    speed = np.sqrt(vx**2 + vy**2)
    # Direction du bateau (si vitesse nulle, on prend 0,0)
    if speed > 0.01:
        dir_boat_x = vx / speed
        dir_boat_y = vy / speed
    else:
        # Si à l'arrêt, on n'a pas vraiment de direction, 
        # mais on peut garder la dernière ou mettre 0
        dir_boat_x, dir_boat_y = 0, 0

    # Feature 2: Vitesse normalisée (0 à 1)
    feat_speed = np.clip(speed / MAX_SPEED, 0, 1)
    
    # --- C. VENT ---
    wind_speed = np.sqrt(wx**2 + wy**2)
    if wind_speed > 0:
        dir_wind_x = wx / wind_speed
        dir_wind_y = wy / wind_speed
    else:
        dir_wind_x, dir_wind_y = 0, 0
        
    # Feature 3: Force du vent (0 à 1)
    feat_wind_str = np.clip(wind_speed / MAX_WIND, 0, 1)
    
    # --- D. RELATIONS (Dot Products - Les features "intelligentes") ---
    
    # Feature 4: Alignement Bateau / But 
    # (Est-ce que je vais vers le but ? 1=oui, -1=dos au but)
    feat_align_goal = (dir_boat_x * dir_goal_x) + (dir_boat_y * dir_goal_y)
    
    # Feature 5: Alignement Vent / But 
    # (Est-ce que le vent pousse vers le but ? 1=vent arrière vers but, -1=vent de face vers but)
    # C'est crucial pour savoir si on doit tirer des bords (tacking)
    feat_wind_goal = (dir_wind_x * dir_goal_x) + (dir_wind_y * dir_goal_y)
    
    # Feature 6: Angle d'incidence du vent sur le bateau (Cos)
    # (Détermine l'efficacité de la voile)
    feat_angle_wind = (dir_boat_x * dir_wind_x) + (dir_boat_y * dir_wind_y)
    
    # Feature 7: Cross Product (Sinus) Bateau / But
    # (Est-ce que le but est à ma gauche ou à ma droite ? Utile pour corriger la trajectoire)
    feat_cross_goal = (dir_boat_x * dir_goal_y) - (dir_boat_y * dir_goal_x)

    # Feature 8: Cross Product (Sinus) Bateau / Vent
    # (Le vent vient-il de babord ou tribord ?)
    feat_cross_wind = (dir_boat_x * dir_wind_y) - (dir_boat_y * dir_wind_x)
    
    # --- E. COMBINAISON ---
    # On retourne un vecteur dense de features toutes bornées ~[-1, 1]
    features = np.array([
        feat_dist,          # Proximité
        feat_speed,         # Cinétique
        feat_wind_str,      # Force vent
        feat_align_goal,    # Cap ok ?
        feat_wind_goal,     # Situation tactique (Vent favorable ?)
        feat_angle_wind,    # Physics (Efficacité)
        feat_cross_goal,    # Correction cap (Gauche/Droite)
        feat_cross_wind,    # Amure (Babord/Tribord)
        dir_boat_x,         # Orientation absolue X
        dir_boat_y,         # Orientation absolue Y
        dir_wind_x,         # Vent absolu X
        dir_wind_y          # Vent absolu Y
    ], dtype=np.float32)
    
    return features



# =============================================================================
# NOISYNET LINEAR LAYER
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Une couche linéaire avec du bruit paramétrique pour l'exploration.
    y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b)
    """
    def __init__(self, in_features, out_features, std_init=1.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Paramètres apprenables (Mu)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Paramètres de bruit apprenables (Sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Buffers pour le bruit (ne sont pas des paramètres du modèle, juste des variables temporaires)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        # Initialisation des poids moyens (comme une couche linéaire normale)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # Initialisation du sigma (le niveau de bruit initial)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    
    def _scale_noise(self, size):
        # Bruit factorisé pour économiser du calcul (Factorized Gaussian Noise)
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Produit extérieur pour générer la matrice de bruit W
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input):
        if self.training:
            # En entraînement : On utilise W = mu + sigma * epsilon
            return F.linear(input, 
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # En évaluation : On coupe le bruit, on utilise juste la moyenne (déterministe)
            return F.linear(input, self.weight_mu, self.bias_mu)


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

class SumTree:
    """
    Structure de données pour PER (accès O(log n) au lieu de O(n))
    """
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
    """
    Buffer de replay avec priorités pour DQN
    
    Args:
        capacity: Taille max du buffer
        alpha: Degré de priorisation (0=uniforme, 1=full priorité)
        beta_start: Correction initiale du biais
        beta_frames: Nombre de frames pour atteindre beta=1.0
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, device='cpu'):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 0.09  # Petit constant pour éviter priorité = 0
        self.abs_err_upper = 1.0  # Clip l'erreur TD
        self.frame = 1
        self.device = device
    
    def push(self, wind_field, physics, action, reward, next_wind_field, next_physics, done):
        """Ajoute une transition avec priorité max"""
        transition = (wind_field, physics, action, reward, next_wind_field, next_physics, done)
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        
        self.tree.add(max_priority, transition)
    
    def sample(self, batch_size):
        """
        Sample un batch avec priorités
        
        Returns:
            wind, physics, actions, rewards, next_wind, next_physics, dones, indices, weights
        """
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        # Incrémente beta progressivement
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calcul des importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        weights /= weights.max()  # Normalisation
        
        self.frame += 1
        
        # Unpack batch
        wind_list, physics_list, actions, rewards, next_wind_list, next_physics_list, dones = zip(*batch)
        
        # Convert to tensors
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
        """Met à jour les priorités après le gradient"""
        for idx, error in zip(indices, td_errors):
            # Priority = (|TD error| + ε)^α
            priority = (abs(error) + self.epsilon) ** self.alpha
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# =============================================================================
# Q-NETWORK AVEC CNN + NOISYNET
# =============================================================================

class QNetworkCNN(nn.Module):
    """
    Q-Network avec CNN pour encoder le wind field + NoisyNet pour exploration.
    
    Architecture:
    - CNN: 32x32x2 → 64 features
    - MLP: 12 physics features → 64 features (avec NoisyLinear)
    - Combine: 128 → 128 → 9 actions (avec NoisyLinear)
    """
    
    def __init__(self, n_physics_features=12, use_noisy_net=True):
        super(QNetworkCNN, self).__init__()
        
        self.use_noisy_net = use_noisy_net
        
        # CNN pour le wind field (reste identique)
        self.wind_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4x4 -> 1x1 (Global Average Pooling)
            nn.Flatten()
        )
        
        # MLP pour les physics features (avec NoisyLinear si activé)
        if use_noisy_net:
            self.physics_mlp = nn.Sequential(
                NoisyLinear(n_physics_features, 64),
                nn.ReLU(),
                NoisyLinear(64, 64),
                nn.ReLU(),
            )
        else:
            self.physics_mlp = nn.Sequential(
                nn.Linear(n_physics_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        
        # Combinaison des features (avec NoisyLinear si activé)
        if use_noisy_net:
            self.combine = nn.Sequential(
                NoisyLinear(128, 128),
                nn.ReLU(),
                NoisyLinear(128, 128), 
                nn.ReLU(),
                NoisyLinear(128, 9)
            )
        else:
            self.combine = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 9)
            )
    
    def forward(self, wind_field, physics):
        """
        Forward pass.
        
        Args:
            wind_field: (batch, 2, 32, 32)
            physics: (batch, n_physics_features)
        
        Returns:
            q_values: (batch, 9)
        """
        # Encode wind field
        wind_features = self.wind_cnn(wind_field)
        wind_features = wind_features.view(wind_features.size(0), -1)  # Flatten
        
        # Encode physics
        physics_features = self.physics_mlp(physics)
        
        # Combine
        combined = torch.cat([wind_features, physics_features], dim=1)
        q_values = self.combine(combined)
        
        return q_values
    
    def reset_noise(self):
        """Reset le bruit de toutes les couches NoisyLinear"""
        if not self.use_noisy_net:
            return
        
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# =============================================================================
# DQN TRAINER AVEC NOISYNET + PER
# =============================================================================

class DQNTrainer:
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
        per_beta_frames=120000,
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
        self.train_scenarios = train_scenarios or ['scenario_1']
        
        # Goal position
        self.goal = (env.goal_position[0], env.goal_position[1])
        
        # Physics features dimension
        n_physics_features = 12
        
        # Créer les réseaux
        self.q_network = QNetworkCNN(
            n_physics_features=n_physics_features,
            use_noisy_net=use_noisy_net
        ).to(device)
        
        self.target_network = QNetworkCNN(
            n_physics_features=n_physics_features,
            use_noisy_net=use_noisy_net
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer avec learning rate decay
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        
        # Replay buffer (PER ou normal)
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_capacity,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                device=device
            )
        else:
            # Fallback to normal replay buffer if needed
            raise NotImplementedError("Normal replay buffer not implemented in this version")
        
        # Logging
        self.writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None
        
        # Compteurs
        self.steps = 0
        self.episodes = 0
        
        # Pour reward shaping
        self.prev_distance = None
        
        print(f"🎮 DQN Trainer initialized:")
        print(f"   NoisyNet: {use_noisy_net}")
        print(f"   PER: {use_per}")
        print(f"   Double DQN: {use_double_dqn}")
        print(f"   Device: {device}")
        print(f"   Goal: {self.goal}")
    
    def compute_shaped_reward(self, obs, next_obs, raw_reward, done):
        """
        Reward shaping pour guider l'apprentissage.
        """
        # Distance au goal
        x, y = next_obs[0], next_obs[1]
        current_dist = np.sqrt((self.goal[0] - x)**2 + (self.goal[1] - y)**2)
        
        # Reward de base
        shaped_reward = raw_reward
        
        # Bonus de progression
        if self.prev_distance is not None:
            progress = self.prev_distance - current_dist
            shaped_reward += progress * 15
        
        self.prev_distance = current_dist
        
        # Bonus de vitesse (encourage le mouvement)
        vx, vy = next_obs[2], next_obs[3]
        speed = np.sqrt(vx**2 + vy**2)
        shaped_reward += speed * 0.9 - 2 # penalise number of step

        if done:
            self.prev_distance = None
        
        return shaped_reward
    
    def get_action(self, obs, greedy=False):
        """
        Sélection d'action avec NoisyNet (pas besoin d'epsilon-greedy).
        """
        # Extract wind field
        wind_field = obs[6:].reshape(32, 32, 2)
        wind_tensor = torch.FloatTensor(wind_field).permute(2, 0, 1).to(self.device).unsqueeze(0)
        
        # Extract physics features
        physics = compute_physics_features(obs, self.goal)
        physics_tensor = torch.FloatTensor(physics).to(self.device).unsqueeze(0)
        
        # Set mode
        if greedy:
            self.q_network.eval()
        else:
            self.q_network.train()
        
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def train_step(self):
        """Un step de training sur un batch avec PER."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch avec PER
        wind, physics, actions, rewards, next_wind, next_physics, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Compute Q(s, a)
        q_values = self.q_network(wind, physics)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target: r + γ * Q_target(s', a')
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: online network selects, target network evaluates
                next_actions = self.q_network(next_wind, next_physics).argmax(1)
                next_q_values = self.target_network(next_wind, next_physics)
                next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Classic DQN: target network does both
                next_q_values = self.target_network(next_wind, next_physics)
                next_q_value = next_q_values.max(1).values
            
            target = rewards + self.gamma * next_q_value * (1 - dones)
        
        # TD errors (pour PER)
        td_errors = (q_value - target).detach().cpu().numpy()
        
        # Loss avec importance sampling weights
        loss = (weights * F.smooth_l1_loss(q_value, target, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities dans PER
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Reset noise pour NoisyNet
        if self.use_noisy_net:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Log loss à TensorBoard
        if self.writer and self.steps % 10 == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), self.steps)
            
            with torch.no_grad():
                mean_q = q_values.mean().item()
                max_q = q_values.max().item()
                self.writer.add_scalar('Train/MeanQValue', mean_q, self.steps)
                self.writer.add_scalar('Train/MaxQValue', max_q, self.steps)
                
                if self.use_per:
                    self.writer.add_scalar('Train/PER_Beta', self.replay_buffer.beta, self.steps)
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.steps}")
        
        return loss.item()
    
    def collect_episode(self, max_steps=200):
        """Collecte un épisode avec reward shaping."""
        obs, _ = self.env.reset()
        episode_reward = 0
        self.prev_distance = None  # Reset pour nouvel épisode
        
        for step in range(max_steps):
            # Get action
            action = self.get_action(obs)
            
            # Step
            next_obs, raw_reward, done, truncated, _ = self.env.step(action)
            
            # Shaped reward
            shaped_reward = self.compute_shaped_reward(obs, next_obs, raw_reward, done or truncated)
            
            # Extraire wind fields et physics
            wind_field = obs[6:].reshape(32, 32, 2)
            next_wind_field = next_obs[6:].reshape(32, 32, 2)
            
            physics = compute_physics_features(obs, self.goal)
            next_physics = compute_physics_features(next_obs, self.goal)
            
            # Push to buffer
            self.replay_buffer.push(
                wind_field, physics, action, shaped_reward,
                next_wind_field, next_physics, done or truncated
            )
            
            episode_reward += shaped_reward
            obs = next_obs
            self.steps += 1
            
            # Train every 4 steps (ratio 1:4)
            if self.steps % 4 == 0:
                loss = self.train_step()
            
            if done or truncated:
                break
        
        self.episodes += 1
        return episode_reward
    
    def evaluate(self, n_episodes=5, max_steps=200):
        """Évalue l'agent (greedy)."""
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
                    if reward > 0:  # Succès
                        successes += 1
                    break
                
                if truncated:
                    break
            
            total_reward += episode_reward
        
        return total_reward / n_episodes, successes / n_episodes
    
    def train(self, num_episodes, eval_freq=100, save_freq=500, verbose=True):
        """
        Boucle d'entraînement principale.
        
        Args:
            num_episodes: Nombre d'épisodes
            eval_freq: Fréquence d'évaluation
            save_freq: Fréquence de sauvegarde
            verbose: Afficher les détails
        """
        best_eval_reward = -np.inf
        
        for episode in range(num_episodes):
            # Curriculum learning
            progress = episode / num_episodes
            init_params, evol_params = generate_curriculum_params(progress)
            
            # Créer environnement avec curriculum
            self.env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
            
            # Collect episode
            episode_reward = self.collect_episode()
            
            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
                self.writer.add_scalar('Train/LearningRate', 
                                      self.scheduler.get_last_lr()[0], episode)
                self.writer.add_scalar('Train/BufferSize', 
                                      len(self.replay_buffer), episode)
                self.writer.add_scalar('Train/Steps', self.steps, episode)
            
            # Log
            if verbose and episode % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"LR: {current_lr:.10f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Steps: {self.steps}")
            
            # Eval
            if episode % eval_freq == 0 and episode > 0:
                eval_reward, success_rate = self.evaluate(n_episodes=10)
                
                if self.writer:
                    self.writer.add_scalar('Eval/Reward', eval_reward, episode)
                    self.writer.add_scalar('Eval/SuccessRate', success_rate, episode)
                
                print(f"[EVAL] Episode {episode} | "
                      f"Eval Reward: {eval_reward:.2f} | "
                      f"Success Rate: {success_rate:.1%}")
            
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'checkpoints/dqn/checkpoint_ep{episode}.pth')
                print(f"💾 Checkpoint saved: episode {episode}")
        
        final_path = 'checkpoints/dqn/final_model.pth'
        self.save_model(final_path)
        print(f"💾 Final model saved: {final_path}")
        
        # Close writer
        if self.writer:
            self.writer.close()
    
    def save_model(self, path='dqn_model.pth'):
        """Sauvegarde le modèle."""
        torch.save(self.q_network.state_dict(), path)