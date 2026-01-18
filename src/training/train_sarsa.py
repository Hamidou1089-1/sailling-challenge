#!/usr/bin/env python3
"""Training script for Q-Learning agent."""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.sarsa import MyAgent
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
    agent = MyAgent(learning_rate=learning_rate, discount_factor=gamma, exploration_rate=0.0, lr_decay=lr_decay)
    np.random.seed(seed)
    agent.seed(seed)
    
    goal = config['environment']['goal']
    
    # Tracking
    rewards_history, steps_history, success_history = agent.train(num_episodes=num_episodes,
                                                                  max_steps=max_steps,
                                                                  goal=goal,
                                                                  min_lr=min_lr) 
    
    print(f"Training Q-Learning for {num_episodes} episodes...")
    print(f"LR: {learning_rate}, Gamma: {gamma}, Decay: {lr_decay}")
    #print(f"Reward shaping - Progress: {prog}, Velocity: {mu}, Step penalty: {step_penalty}\n")
    
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
    output_path = f"src/submission/my_agent_sarsa_{exp_name}.py"
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