import numpy as np
import sys
import os

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

from torch.optim.lr_scheduler import ExponentialLR

import pickle
from wind_scenarios.sailing_physics import calculate_sailing_efficiency 


# =============================================================================
# COLLECTE DES STATISTIQUES DE NORMALISATION
# =============================================================================




def compute_physics_features_raw(obs: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """
    Calcule les physics features SANS normalisation (pour collecte de stats).
    
    Returns:
        Array de 15 features brutes
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    wind_field = obs[6:].reshape(32, 32, 2)
    
    # Vecteurs
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
    
    # Alignement vent-goal
    if wind_magnitude > 0.1 and distance_to_goal > 0:
        wind_goal_alignment = np.dot(wind_vec, goal_vec) / (wind_magnitude * distance_to_goal)
    else:
        wind_goal_alignment = 0.0
    
    # Angle relatif vitesse-vent
    if velocity_magnitude > 0.1 and wind_magnitude > 0.1:
        v_angle = np.arctan2(vy, vx)
        w_angle = np.arctan2(wy, wx)
        relative_wind_angle = v_angle - w_angle
        relative_wind_angle = np.arctan2(np.sin(relative_wind_angle), np.cos(relative_wind_angle))
    else:
        relative_wind_angle = 0.0
    
    # Équations cinématiques (physique réelle simplifiée)
    # Accélération due au vent (coefficient simplifié)
    # Dans un vrai cas, utiliser calculate_sailing_efficiency
    dt = 1.0
    wind_efficiency = 0.4  # Coefficient moyen (devrait venir de sailing_physics)
    ax = wind_efficiency * wx
    ay = wind_efficiency * wy
    
    # Prédiction position
    predicted_x = x + vx * dt + 0.5 * ax * dt**2
    predicted_y = y + vy * dt + 0.5 * ay * dt**2
    predicted_goal_vec = np.array(goal) - np.array([predicted_x, predicted_y])
    predicted_distance = np.linalg.norm(predicted_goal_vec)
    
    # Amélioration prédite
    predicted_improvement = distance_to_goal - predicted_distance
    
    # Wind ahead
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
    
    # Wind asymmetry (gauche vs droite)
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
    
    # Assembler (15 features)
    physics_features = np.array([
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
    
    return physics_features


def compute_physics_features(obs: np.ndarray, goal: Tuple[int, int], 
                            feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
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
    # Calculer features brutes
    features_raw = compute_physics_features_raw(obs, goal)
    
    # Normaliser avec mean/std
    features_norm = (features_raw - feature_mean) / (feature_std + 1e-16)
    
    return features_norm



def collect_normalization_stats(env, n_episodes=500, save_path='normalization_stats.pkl'):
    """
    Collecte les statistiques (mean, std) des physics features.
    À exécuter UNE FOIS avant l'entraînement.
    
    Args:
        env: Environnement sailing
        n_episodes: Nombre d'épisodes pour collecter les stats
        save_path: Où sauvegarder les stats
    
    Returns:
        (feature_mean, feature_std): Arrays numpy
    """
    print(f"Collecting normalization statistics over {n_episodes} episodes...")
    
    all_features = []
    goal = (16, 31)
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        
        for step in range(200):
            # Calculer les physics features SANS normalisation
            features = compute_physics_features_raw(obs, goal)
            all_features.append(features)
            
            # Action random
            action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)
            
            if done or truncated:
                break
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")
    
    all_features = np.array(all_features)
    
    # Calculer mean et std
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    
    # Éviter division par zéro
    feature_std = np.where(feature_std < 1e-14, 1.0, feature_std)
    
    # Sauvegarder
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': feature_mean, 'std': feature_std}, f)
    
    print(f"✓ Stats saved to {save_path}")
    print(f"Feature means: {feature_mean}")
    print(f"Feature stds: {feature_std}")
    
    return feature_mean, feature_std



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
# AGENT DQN 
# =============================================================================

class MyAgentDQN(BaseAgent):
    """
    Agent DQN pour Sailing Challenge.
    Compatible avec BaseAgent pour evaluation/submission.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 stats_path: str = 'normalization_stats.pkl',
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
        
        # Charger les stats de normalisation
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.feature_mean = stats['mean']
            self.feature_std = stats['std']
        
        # Créer le réseau
        self.q_network = QNetworkCNN(n_physics_features=15).to(self.device)
        
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
        physics = compute_physics_features(observation, self.goal, 
                                          self.feature_mean, self.feature_std)
        
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
        self.physics = np.zeros((capacity, 15), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_wind_fields = np.zeros((capacity, 32, 32, 2), dtype=np.float32)
        self.next_physics = np.zeros((capacity, 15), dtype=np.float32)
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
    
    def __init__(self, env, stats_path, learning_rate=1e-3, lr_decay=.9998,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 gamma=0.99, buffer_capacity=100000, batch_size=64,
                 learning_starts=1000, target_update_freq=1000,
                 gradient_clip=10.0, device='cpu', use_double_dqn=True):
        
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
        
        # Charger les stats de normalisation
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.feature_mean = stats['mean']
            self.feature_std = stats['std']
        
        # Networks
        self.q_network = QNetworkCNN(n_physics_features=15).to(self.device)
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
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'goal': self.goal,
            'eval_results': eval_results
        }
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath, env):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        # Create trainer (will create new networks)
        trainer = cls(env, stats_path=None, device='cpu')
        
        # Load states
        trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epsilon = checkpoint['epsilon']
        trainer.episodes = checkpoint['episode']
        trainer.steps = checkpoint['steps']
        trainer.feature_mean = checkpoint['feature_mean']
        trainer.feature_std = checkpoint['feature_std']
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
            progress_reward = 9.0 * progress  # Coefficient fort
        else:
            progress_reward = 0.0
        
        self.prev_distance = distance
        
        # Velocity reward
        velocity = np.linalg.norm([vx, vy])
        velocity_reward = 0.39 * velocity
        
        # Malus de step
        step_penalty = -0.5
        
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
        physics = compute_physics_features(obs, self.goal, 
                                          self.feature_mean, self.feature_std)
        
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
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        self.scheduler.step()
        
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
            
            physics = compute_physics_features(obs, self.goal, 
                                              self.feature_mean, self.feature_std)
            next_physics = compute_physics_features(next_obs, self.goal,
                                                   self.feature_mean, self.feature_std)
            
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
            # Collect episode
            episode_reward = self.collect_episode()
            
            # Log
            if verbose and episode % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Steps: {self.steps}")
            
            # Eval
            if episode % eval_freq == 0 and episode > 0:
                eval_reward, success_rate = self.evaluate(n_episodes=5)
                print(f"[EVAL] Episode {episode} | "
                      f"Eval Reward: {eval_reward:.2f} | "
                      f"Success Rate: {success_rate:.1%}")
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_model('dqn_best_model.pth')
                    print(f"✓ New best model saved! (Reward: {eval_reward:.2f})")
            
            # Save checkpoint
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'dqn_checkpoint_{episode}.pth')
    
    def save_model(self, path='dqn_model.pth'):
        """Sauvegarde le modèle."""
        torch.save(self.q_network.state_dict(), path)







