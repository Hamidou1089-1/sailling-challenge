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
#     if np.random.random() < 0.3:
#         difficulty = 0.0  # Mode "Repos / Fondamentaux"
#     else:
#         # La difficulté suit la progression. 
#         # On ajoute un petit bruit pour ne pas être trop linéaire.
#         difficulty = np.clip(progress + np.random.uniform(-1, 1), 0.0, 1.0)

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
#         'pattern_strength': 0.1 + (0.5 * difficulty),
#         'strength_variation': 0.08 + (0.5 * difficulty),
#         'noise': 0.085 + (0.05 * difficulty)
#     }
    
#     # --- 4. EVOLUTION DYNAMIQUE ---
#     wind_evol_params = {
#         # Probabilité de changement : De 0% (Stable) à 90% (Chaos)
#         'wind_change_prob': 0.15 + (0.75 * difficulty),
#         'pattern_scale': 128,
#         'perturbation_angle_amplitude': 0.085 + (0.15 * difficulty),
#         'perturbation_strength_amplitude': 0.085 + (0.15 * difficulty),
        
        
#         'rotation_bias': np.random.uniform(-0.045, 0.045) * difficulty,
#         'bias_strength': np.clip(difficulty + 0.15, 0, 1.0) 
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
    if np.random.random() < 0.3:
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
    MAX_SPEED = 7.0  # Vitesse max raisonnable du bateau
    MAX_WIND = 7.0   # Vitesse max du vent
    
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
# Q-NETWORK AVEC CNN
# =============================================================================

class QNetworkCNN(nn.Module):
    """
    Q-Network avec CNN pour encoder le wind field.
    
    Architecture:
    - CNN: 32x32x2 → 64 features
    - MLP: 15 physics features → 64 features
    - Combine: 128 → 128 → 9 actions
    """
    
    def __init__(self, n_physics_features=12):
        super(QNetworkCNN, self).__init__()
        
        self.wind_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # 32×32 → 16×16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16×16 → 8×8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8×8 → 4×4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4×4×64 → 1×1×64 (GAP)
            nn.Flatten()
            #nn.Linear(64, 1) 
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
# AGENT DQN 
# =============================================================================

class MyAgentDQN(BaseAgent):
    """
    Agent DQN pour Sailing Challenge.
    Compatible avec BaseAgent pour evaluation/submission.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device='cpu'):
        """
        Args:
            model_path: Chemin vers le modèle sauvegardé (.pth)
            stats_path: Chemin vers les stats de normalisation (.pkl)
            device: 'cpu' ou 'cuda'
        """
        super().__init__()
        self.device = torch.device(device)
        self.goal = (16, 31)
        
        
        
        # Créer le réseau
        self.q_network = QNetworkCNN(n_physics_features=12).to(self.device)
        
        # Charger le modèle si fourni
        if model_path:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.q_network.eval()
    
    def act(self, observation: np.ndarray) -> int:
        """
        Sélectionne l'action avec la plus grande Q-value.
        
        Args:
            observation: État brut [x, y, vx, vy, wx, wy, wind_field...]
        
        Returns:
            action: Entier entre 0 et 8
        """
        # Extraire wind field
        wind_field = observation[6:].reshape(32, 32, 2)
        
        # Calculer physics features normalisées
        physics = compute_physics_features(observation, self.goal)
        
        # Convertir en tensors
        wind_tensor = torch.tensor(wind_field.transpose(2, 0, 1), 
                                   dtype=torch.float32, device=self.device).unsqueeze(0)
        physics_tensor = torch.tensor(physics, dtype=torch.float32, 
                                     device=self.device).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def reset(self):
        """Reset (rien à faire pour DQN)."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)




# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Replay buffer optimisé avec wind field séparé."""
    
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Stockage séparé pour économiser la mémoire
        self.wind_fields = np.zeros((capacity, 32, 32, 2), dtype=np.float32)
        self.physics = np.zeros((capacity, 12), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_wind_fields = np.zeros((capacity, 32, 32, 2), dtype=np.float32)
        self.next_physics = np.zeros((capacity, 12), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, wind_field, physics, action, reward, next_wind_field, next_physics, done):
        """Ajoute une transition."""
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
        """Sample un batch aléatoire."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Wind fields: (batch, 32, 32, 2) → (batch, 2, 32, 32) pour PyTorch
        wind_fields = torch.tensor(
            self.wind_fields[indices].transpose(0, 3, 1, 2), 
            device=self.device
        )
        next_wind_fields = torch.tensor(
            self.next_wind_fields[indices].transpose(0, 3, 1, 2),
            device=self.device
        )
        
        physics = torch.tensor(self.physics[indices], device=self.device)
        actions = torch.tensor(self.actions[indices], device=self.device)
        rewards = torch.tensor(self.rewards[indices], device=self.device)
        next_physics = torch.tensor(self.next_physics[indices], device=self.device)
        dones = torch.tensor(self.dones[indices], device=self.device)
        
        return wind_fields, physics, actions, rewards, next_wind_fields, next_physics, dones
    
    def __len__(self):
        return self.size




# =============================================================================
# TRAINER 
# =============================================================================

class DQNTrainer:
    """
    Classe pour entraîner l'agent DQN avec toutes les améliorations.
    """
    
    def __init__(self, env,  learning_rate=1e-3, lr_decay=.9998,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 gamma=0.99, buffer_capacity=100000, batch_size=64,
                 learning_starts=1000, target_update_freq=1000,
                 gradient_clip=10.0, device='cpu', use_double_dqn=True, 
                 train_scenarios=['training_1', 'training_2', 'training_3' ], tensorboard_dir=None):
        
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.goal = (16, 31)

        self.learning_starts = learning_starts
        self.gradient_clip = gradient_clip
        
        
        
        # Networks
        self.q_network = QNetworkCNN(n_physics_features=12).to(self.device)
        self.target_network = deepcopy(self.q_network)
        
        # Optimizer avec scheduler
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.device)
        
        # Metrics
        self.steps = 0
        self.episodes = 0
        self.prev_distance = None

        self.use_double_dqn = use_double_dqn 
        self.train_scenarios = train_scenarios 

        self.writer = None
        if tensorboard_dir:
            self.writer = SummaryWriter(tensorboard_dir)
            print(f"📊 TensorBoard logging to: {tensorboard_dir}")
        
        print(f"Using Double DQN: {self.use_double_dqn}") 
    
    def save_checkpoint(self, filepath, eval_results=None):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episodes,
            'steps': self.steps,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'goal': self.goal,
            'eval_results': eval_results
        }
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath, env):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        # Create trainer (will create new networks)
        trainer = cls(env, device='cpu')
        
        # Load states
        trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epsilon = checkpoint['epsilon']
        trainer.episodes = checkpoint['episode']
        trainer.steps = checkpoint['steps']
        trainer.goal = checkpoint['goal']
        
        return trainer
    
    def compute_shaped_reward(self, obs, next_obs, raw_reward, done):
        """
        Reward shaping rigoureux.
        
        Returns:
            shaped_reward: Reward amélioré
        """
        x, y = next_obs[0], next_obs[1]
        vx, vy = next_obs[2], next_obs[3]
        
        # Distance au goal
        goal_vec = np.array(self.goal) - np.array([x, y])
        distance = np.linalg.norm(goal_vec)
        
        # Progress reward
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            progress_reward = 10.0 * progress  
        else:
            progress_reward = 0.0
        
        self.prev_distance = distance
        
        # Velocity reward
        velocity = np.linalg.norm([vx, vy])
        velocity_reward = 0.9 * velocity
        
        # Malus de step
        step_penalty = -1.5
        
        # Shaped reward
        shaped_reward = raw_reward + progress_reward + velocity_reward + step_penalty
        
        # Reset prev_distance si épisode terminé
        if done:
            self.prev_distance = None
        
        return shaped_reward
    
    def get_action(self, obs, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(9)
        
        # Extraire wind field et physics
        wind_field = obs[6:].reshape(32, 32, 2)
        physics = compute_physics_features(obs, self.goal)
        
        # Tensors
        wind_tensor = torch.tensor(wind_field.transpose(2, 0, 1), 
                                   dtype=torch.float32, device=self.device).unsqueeze(0)
        physics_tensor = torch.tensor(physics, dtype=torch.float32, 
                                     device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(wind_tensor, physics_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        """Un step de training sur un batch."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        wind, physics, actions, rewards, next_wind, next_physics, dones = \
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
        
        # Loss
        loss = F.smooth_l1_loss(q_value, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

        # Log loss à TensorBoard ← AJOUT
        if self.writer and self.steps % 10 == 0:  # Log toutes les 10 steps
            self.writer.add_scalar('Train/Loss', loss.item(), self.steps)
            
            # Log Q-values moyennes (optionnel mais utile)
            with torch.no_grad():
                mean_q = q_values.mean().item()
                max_q = q_values.max().item()
                self.writer.add_scalar('Train/MeanQValue', mean_q, self.steps)
                self.writer.add_scalar('Train/MaxQValue', max_q, self.steps)
        
        
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
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
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

            progress = episode / num_episodes
            init_params, evol_params = generate_curriculum_params(progress)
                    
            

            self.env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
            # Collect episode
            episode_reward = self.collect_episode()

            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
                self.writer.add_scalar('Train/Epsilon', self.epsilon, episode)
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
                      f"Epsilon: {self.epsilon:.5f} | "
                      f"LR: {current_lr:.10f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Steps: {self.steps}")
            
            # Eval
            if episode % eval_freq == 0 and episode > 0:
                eval_reward, success_rate = self.evaluate(n_episodes=5)

                if self.writer:
                    self.writer.add_scalar('Eval/Reward', eval_reward, episode)
                    self.writer.add_scalar('Eval/SuccessRate', success_rate, episode)
            
                print(f"[EVAL] Episode {episode} | "
                      f"Eval Reward: {eval_reward:.2f} | "
                      f"Success Rate: {success_rate:.1%}")
                
                # # Save best model
                # if eval_reward > best_eval_reward:
                #     best_eval_reward = eval_reward
                #     self.save_model('dqn_best_model.pth')
                #     print(f"✓ New best model saved! (Reward: {eval_reward:.2f})")
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'checkpoints/dqn/checkpoint_ep{episode}.pth')
                print(f"💾 Checkpoint saved: episode {episode}")
              
            # # Save checkpoint
            # if episode % save_freq == 0 and episode > 0:
            #     self.save_model(f'dqn_checkpoint_{episode}.pth')
        
        final_path = 'checkpoints/dqn/final_model.pth'
        self.save_model(final_path)
        print(f"💾 Final model saved: {final_path}")
        
        # Close writer ← AJOUT
        if self.writer:
            self.writer.close()

    def save_model(self, path='dqn_model.pth'):
        """Sauvegarde le modèle."""
        torch.save(self.q_network.state_dict(), path)






