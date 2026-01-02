import numpy as np

from .base_agent import BaseAgent
from typing import Dict, Any, Tuple, Optional

class MyAgent_n_steps(BaseAgent):

    def __init__(self, goal=[16, 31], learning_rate=0.1, discount_factor=0.9, exploration_rate=0.0, n_steps=3):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        
        
        self.goal = goal

        self.weather_cache = None
        self.last_observation_id = None
        self.angle_bins = np.linspace(-np.pi, np.pi, 12)

        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}

        self.action_counts = {}  # Combien de fois chaque (state, action)
        self.total_visits = {} 

        self.n_steps = n_steps
        self.buffer = []  # Buffer des n derniers steps

    
    
    
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
        
        # Velocity features
        v_magnitude = np.linalg.norm([vx, vy])
        if v_magnitude < 1.0:
            velocity_magnitude_bin = 0
        elif v_magnitude < 3.0:
            velocity_magnitude_bin = 1
        elif v_magnitude < 6.0:
            velocity_magnitude_bin = 2
        else:
            velocity_magnitude_bin = 3
        
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
            wind_asymmetry_bin
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

        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(9)
            self.total_visits[state] = 0
        
        # # Epsilon-greedy action selection
        # if self.np_random.random() < self.exploration_rate:
        #     # Explore: choose a random action
        #     return self.np_random.integers(0, 9)
        # else:
        #     # Exploit: choose the best action according to Q-table
        #     if state not in self.q_table:
        #         # If state not in Q-table, initialize it
        #         self.q_table[state] = np.zeros(9)
            
        #     # Return action with highest Q-value
        #     return np.argmax(self.q_table[state])
        return self.ucb_select(state)
    
    def ucb_select(self, state, c=2.0):
        """UCB exploration."""
        ucb_values = []
        total_visits = self.total_visits[state]
        
        for action in range(9):
            n_action = self.action_counts[state][action]
            
            if n_action == 0:
                # Action jamais essayée → priorité maximale
                return action
            
            # Q-value + bonus d'exploration
            q_value = self.q_table[state][action] if state in self.q_table else 0
            exploration_bonus = c * np.sqrt(np.log(total_visits + 1) / n_action)
            
            ucb_values.append(q_value + exploration_bonus)
        
        return np.argmax(ucb_values)
    
    def ucrl2_select(self, state, delta=0.05):
        """UCRL2 : UCB + confiance statistique."""
        ucb_values = []
        
        for action in range(9):
            n = self.action_counts[state][action]
            
            if n == 0:
                return action
            
            # Borne de confiance
            confidence_radius = np.sqrt(14 * np.log(2 / delta) / n)
            
            # Q optimiste
            q_optimistic = self.q_table[state][action] + confidence_radius
            ucb_values.append(q_optimistic)
        
        return np.argmax(ucb_values)
    

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table based on observed transition."""
        # Initialize Q-values if states not in table

        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # Si buffer plein OU épisode terminé, faire update
        if len(self.buffer) >= self.n_steps or done:
            self._n_step_update()

        # Si done, vider tout le buffer
        if done:
            while len(self.buffer) > 0:
                self._n_step_update()

        
        
        
    
    def _n_step_update(self):
        if len(self.buffer) == 0:
            return
        
        first = self.buffer[0]
        state_0 = first['state']
        action_0 = first['action']

         
        if state_0 not in self.action_counts:
            self.action_counts[state_0] = np.zeros(9)
            self.total_visits[state_0] = 0
        
        self.action_counts[state_0][action_0] += 1
        self.total_visits[state_0] += 1

        n_step_return = 0
        gamma_power = 1

        for transition in self.buffer:
            n_step_return += gamma_power * transition['reward']
            gamma_power *= self.discount_factor

            if transition['done']:
                break
        
         # Bootstrap si pas terminal
        last = self.buffer[-1]
        if not last['done']:
            last_state = last['next_state']
            if last_state not in self.q_table:
                self.q_table[last_state] = np.zeros(9)
            n_step_return += gamma_power * np.max(self.q_table[last_state])
        
        # Update Q-table
        if state_0 not in self.q_table:
            self.q_table[state_0] = np.zeros(9)
        
        td_error = n_step_return - self.q_table[state_0][action_0]
        self.q_table[state_0][action_0] += self.learning_rate * td_error
        
        # Retirer le premier élément
        self.buffer.pop(0)

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

    
    





