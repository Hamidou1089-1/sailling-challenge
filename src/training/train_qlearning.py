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
        scenario = np.random.choice(train_scenarios)
        env = SailingEnv(**get_wind_scenario(scenario))
        
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