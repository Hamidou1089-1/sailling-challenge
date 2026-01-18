#!/usr/bin/env python3
"""Training script for Q-Learning agent."""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.my_agent import MyAgent
from src.utils.agent_utils import save_my_agent
from wind_scenarios import get_wind_scenario
from wind_scenarios.env_sailing import SailingEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Learning agent")
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()




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
        difficulty = np.clip(progress + np.random.uniform(-0.1, 0.1), 0.0, 1.0)

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


def train_qlearning(config):
    """Training loop for Q-Learning with UCB."""
    num_episodes = config['training']['num_episodes']
    max_steps = config['training']['max_steps_per_episode']
    seed = config['training']['seed']
    train_scenarios = config['training']['train_scenarios']
    
    # Agent params
    learning_rate = config['agent']['learning_rate']
    gamma = config['agent']['gamma']
    lr_decay = config['agent']['lr_decay']
    min_lr = config['agent']['min_lr']
    
    # Reward shaping
    prog = config['agent']['progress_reward_coeff']
    mu = config['agent']['velocity_reward_coeff']
    step_penalty = config['agent']['step_penalty']
    
    # Create agent
    agent = MyAgent(learning_rate=learning_rate, discount_factor=gamma, exploration_rate=0.0)
    np.random.seed(seed)
    agent.seed(seed)
    
    goal = config['environment']['goal']
    
    # Tracking
    rewards_history = []
    steps_history = []
    success_history = []
    
    print(f"Training Q-Learning for {num_episodes} episodes...")
    print(f"LR: {learning_rate}, Gamma: {gamma}, Decay: {lr_decay}")
    print(f"Reward shaping - Progress: {prog}, Velocity: {mu}, Step penalty: {step_penalty}\n")
    
    for episode in range(num_episodes):
        # Random scenario
        #scenario = np.random.choice(train_scenarios)
        progress = episode / num_episodes
        init_params, evol_params = generate_curriculum_params(progress)
                    
            

        env = SailingEnv(wind_init_params=init_params, wind_evol_params=evol_params)
        #env = SailingEnv(**get_wind_scenario(scenario))
        
        observation, info = env.reset(seed=episode)
        state = agent.discretize_state(observation)
        
        total_reward = 0
        x_prev, y_prev = observation[0], observation[1]
        distance_prev = np.sqrt((goal[0]-x_prev)**2 + (goal[1]-y_prev)**2)
        
        for step in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Reward shaping
            x, y = next_observation[0], next_observation[1]
            vx, vy = next_observation[2], next_observation[3]
            
            distance_curr = np.sqrt((goal[0]-x)**2 + (goal[1]-y)**2)
            progress = distance_prev - distance_curr
            progress_reward = prog * progress
            
            velocity = np.sqrt(vx**2 + vy**2)
            velocity_reward = mu * velocity
            
            shaped_reward = progress_reward + velocity_reward + reward + step_penalty
            
            next_state = agent.discretize_state(next_observation)
            agent.learn(state, action, shaped_reward, next_state)
            
            state = next_state
            observation = next_observation
            total_reward += shaped_reward
            distance_prev = distance_curr
            
            if done or truncated:
                break
        
        rewards_history.append(total_reward)
        steps_history.append(step+1)
        success_history.append(done)
        
        # LR decay
        agent.learning_rate = max(min_lr, agent.learning_rate * lr_decay)
        
        # Progress
        if (episode + 1) % 100 == 0:
            recent_success = sum(success_history[-100:]) / 100 * 100
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Success (last 100): {recent_success:.1f}% | "
                  f"LR: {agent.learning_rate:.6f} | "
                  f"Q-table size: {len(agent.q_table)}")
    
    success_rate = sum(success_history) / len(success_history) * 100
    print(f"\n✅ Training complete!")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Avg reward: {np.mean(rewards_history):.2f}")
    print(f"Avg steps: {np.mean(steps_history):.1f}")
    print(f"Q-table size: {len(agent.q_table)} states\n")
    
    return agent, rewards_history, steps_history, success_history

def main():
    args = parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {args.config}\n")
    
    # Create directories
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path('src/submission').mkdir(parents=True, exist_ok=True)
    
    # Train
    agent, rewards, steps, success = train_qlearning(config)
    
    # Save
    exp_name = config['logging']['experiment_name']
    output_path = f"src/submission/my_agent_qlearning_{exp_name}.py"
    save_my_agent(agent, output_path)
    
    # Checkpoint Q-table
    import pickle
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    qtable_path = checkpoint_dir / 'qtable_final.pkl'
    with open(qtable_path, 'wb') as f:
        pickle.dump(agent.q_table, f)
    
    print(f"📁 Submission: {output_path}")
    print(f"💾 Q-table: {qtable_path}")

if __name__ == '__main__':
    main()