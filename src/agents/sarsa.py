import numpy as np

from .base_agent import BaseAgent
from typing import Dict, Any, Tuple, Optional

import numpy as np
from pathlib import Path
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.agent_utils import save_my_agent
from wind_scenarios import get_wind_scenario
from wind_scenarios.env_sailing import SailingEnv




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
        difficulty = np.clip(progress + np.random.uniform(-0.3, 0.3), 0.0, 1.0)

    # --- 3. PARAMÈTRES DU VENT ---
    
    # Vitesse : 3.0 est la vitesse standard. 
    # Plus c'est dur, plus on s'éloigne de cette norme (vent très faible ou tempête).
    # difficulty 0 -> speed 3.0
    # difficulty 1 -> speed entre 1.0 et 5.0
    #speed_noise = np.random.uniform(-.01, .01) * difficulty
    base_speed = 3.0 
    
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



class MyAgent(BaseAgent):

    def __init__(self, goal=[16, 31], learning_rate=0.1, discount_factor=0.99, exploration_rate=0.9999, 
                 lr_decay=0.999):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.lr_decay = lr_decay
        
        
        
        self.goal = goal

        self.weather_cache = None
        self.last_observation_id = None
        self.angle_bins = np.linspace(-np.pi, np.pi, 12)

        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}

        
      
        

    
    
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
            x, y, angle_to_goal, weather, depth=6
        )


        wind_ahead_bin = self.discretize_wind_ahead(alignment, threshold=0.7)
    
        # Calculer wind_asymmetry
        asymmetry_diff = self.compute_wind_asymmetry(
            x, y, angle_to_goal, weather, depth=6
        )
        wind_asymmetry_bin = self.discretize_asymmetry(asymmetry_diff, threshold=0.7)
        
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
        # Au lieu de 8 bins, utilise 12 bins plus fins
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
        goal_dir = np.array([np.cos(angle_to_goal), np.sin(angle_to_goal)])
        
        # Générer toutes les positions à checker d'un coup
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
        winds = weather[valid_positions[:, 0], valid_positions[:, 1]]  # shape (N, 2)
        avg_wind = winds.mean(axis=0)
        
        # Dot product
        return np.dot(avg_wind, goal_dir)
    
    def discretize_wind_ahead(self, alignment, threshold=0.5):
        """
        Discrétise l'alignement en 3 bins.
        """
        if alignment < -threshold:
            return 0  # défavorable (vent contre)
        elif alignment < threshold:
            return 1  # neutre
        else:
            return 2  # favorable (vent pousse vers goal)
    
    def compute_wind_asymmetry(self, x, y, angle_to_goal, weather, depth=2):
        angle_left = angle_to_goal + np.pi/2
        angle_right = angle_to_goal - np.pi/2
        
        wind_left = self._get_wind_vectorized(x, y, angle_left, weather, depth)
        wind_right = self._get_wind_vectorized(x, y, angle_right, weather, depth)
        
        return np.linalg.norm(wind_left) - np.linalg.norm(wind_right)

    def _get_wind_vectorized(self, x, y, angle, weather, depth):
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

    def get_wind_in_direction(self, x, y, angle, weather, depth):
        """
        Helper : extrait le vent moyen dans une direction.
        """
        dir_x = np.cos(angle)
        dir_y = np.sin(angle)
        
        wind_sum_x = 0
        wind_sum_y = 0
        count = 0
        
        for d in range(1, depth + 1):
            check_x = int(x + d * dir_x)
            check_y = int(y + d * dir_y)
            
            if 0 <= check_x < 32 and 0 <= check_y < 32:
                wx, wy = weather[check_x][check_y]
                wind_sum_x += wx
                wind_sum_y += wy
                count += 1
        
        if count == 0:
            return 0, 0
        
        return wind_sum_x / count, wind_sum_y / count

    def discretize_asymmetry(self, diff, threshold=0.3):
        """
        Discrétise la différence gauche-droite.
        """
        if diff > threshold:
            return 0  # gauche meilleur
        elif diff < -threshold:
            return 2  # droite meilleur
        else:
            return 1  # symétrique
        

    def _reconstruct_weather(self, wind):
        """Reconstruit la grille de vent une fois."""
        weather = np.zeros((32, 32, 2))
        for i in range(32):
            for j in range(32):
                idx = (i * 32 + j) * 2
                weather[i, j] = [wind[idx], wind[idx+1]]
        return weather


    def act(self, observation):
        """Choose an action using epsilon-greedy policy."""
        # Discretize the state
        state = self.discretize_state(observation)

        
        # Epsilon-greedy action selection
        if self.np_random.random() < self.exploration_rate:
            # Explore: choose a random action
            return self.np_random.integers(0, 9)
        else:
            # Exploit: choose the best action according to Q-table
            if state not in self.q_table:
                # If state not in Q-table, initialize it
                self.q_table[state] = np.zeros(9)
            
            # Return action with highest Q-value
            return np.argmax(self.q_table[state])
        
    
    

    def learn(self, state, action, reward, next_state, next_action=None):
        """
        Expected Sarsa
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        q_next = self.q_table[next_state]
        q_max = np.max(q_next)

        greedy_actions = np.sum(q_next == q_max)
        non_greedy_prob = self.exploration_rate / 9
        greedy_prob = ((1 - self.exploration_rate) / greedy_actions) + non_greedy_prob


        expected_q = 0
        for a in range(9):
            if q_next[a] == q_max:
                expected_q += q_next[a] * greedy_prob
            else:
                expected_q += q_next[a] * non_greedy_prob

        td_target = reward + self.discount_factor * expected_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def train(self, num_episodes, max_steps, goal, min_lr):
        
        rewards_history = []
        steps_history = []
        success_history = []

        for episode in range(num_episodes):
            progress = episode / num_episodes
            init_params, evol_params = generate_curriculum_params(progress)
                        
                

            env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
            #env = SailingEnv(**get_wind_scenario(scenario))
            
            observation, info = env.reset(seed=episode)
            state = self.discretize_state(observation)
            
            total_reward = 0
            x_prev, y_prev = observation[0], observation[1]
            vx_pre, vy_prev = observation[2], observation[3]
            distance_prev = np.sqrt((goal[0]-x_prev)**2 + (goal[1]-y_prev)**2)
            velocity_prev = np.sqrt(vx_pre**2 +  vy_prev**2)
            
            for step in range(max_steps):
                action = self.act(observation)
                next_observation, reward, done, truncated, info = env.step(action)
                
                # Reward shaping
                x, y = next_observation[0], next_observation[1]

                vx, vy = next_observation[2], next_observation[3]
                
                distance_curr = np.sqrt((goal[0]-x)**2 + (goal[1]-y)**2)
                progress = distance_prev - distance_curr
                progress_reward = 15 * progress
                
                velocity = np.sqrt(vx**2 + vy**2)
                velocity_prog = velocity - velocity_prev
                velocity_reward = 1.5 * velocity_prog 
                
                shaped_reward = progress_reward + velocity_reward + reward - 0.05 * step
                
                next_state = self.discretize_state(next_observation)
                next_action = self.act(next_observation)
                self.learn(state, action, shaped_reward, next_state, next_action)
                
                state = next_state
                observation = next_observation
                action = next_action
                total_reward += shaped_reward
                distance_prev = distance_curr
                velocity_prev = velocity
                
                if done or truncated:
                    break
            
            rewards_history.append(total_reward)
            steps_history.append(step+1)
            success_history.append(done)
            
            # LR decay
            self.learning_rate = max(min_lr, self.learning_rate * self.lr_decay)
            self.exploration_rate = max(0.003, self.exploration_rate * 0.9998 )
            # Progress
            if (episode + 1) % 100 == 0:
                recent_success = sum(success_history[-100:]) / 100 * 100
                print(f"Episode {episode+1}/{num_episodes} | "
                    f"Success: {recent_success:.1f}% | "
                    f"Reward: {sum(rewards_history[-100:]) / 100:.4} | "
                    f"LR: {self.learning_rate:.6f} | "
                    f"Steps: {int(sum(steps_history[-100:]) / 100)}")
    
        return rewards_history, steps_history, success_history


    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset for Q-learning agent
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
        
    def save(self, path):
        """Save the Q-table to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, path):
        """Load the Q-table from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    
    





