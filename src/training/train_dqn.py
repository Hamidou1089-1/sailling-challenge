#!/usr/bin/env python3
"""Training script for DQN agent."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.my_agent_DQN import DQNTrainer, collect_normalization_stats
from src.utils.save_my_dqn import save_dqn_agent
from wind_scenarios import get_wind_scenario
from wind_scenarios.env_sailing import SailingEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--collect-stats', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    
    # Create directories
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path('src/submission').mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(parents=True, exist_ok=True)
    
    stats_path = Path(config['normalization']['stats_path'])
    train_scenarios = config['training']['train_scenarios']
    
    # Collect stats
    if args.collect_stats:
        print(f"\nCollecting normalization stats...")
        #env = SailingEnv(**get_wind_scenario(train_scenarios[0]))
        collect_normalization_stats(
            SailingEnv=SailingEnv, 
            n_episodes=config['normalization']['collect_n_episodes'],
            save_path=str(stats_path),
            train_scenarios=train_scenarios
        )
        print(f"Stats saved to {stats_path}")
        return
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found at {stats_path}. Run with --collect-stats first.")
    
    print(f"Using stats: {stats_path}")
    print(f"Training scenarios: {train_scenarios}")
    
    # Create environment
    env = SailingEnv(**get_wind_scenario(train_scenarios[0]))
    exp_name = config['logging']['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = f"runs/dqn/{exp_name}_{timestamp}"
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    print(f"📊 TensorBoard dir: {tb_dir}")
    
    # Create trainer
    print("\nCreating DQN trainer...")
    trainer = DQNTrainer(
        env,
        stats_path=str(stats_path),
        learning_rate=config['agent']['learning_rate'],
        lr_decay=config['agent'].get('lr_decay', 1.0),
        epsilon_start=config['agent']['epsilon_start'],
        epsilon_end=config['agent']['epsilon_end'],
        epsilon_decay=config['agent']['epsilon_decay'],
        gamma=config['agent']['gamma'],
        buffer_capacity=config['agent']['buffer_capacity'],
        batch_size=config['agent']['batch_size'],
        target_update_freq=config['agent']['target_update_freq'],
        use_double_dqn=config['agent'].get('use_double_dqn', True),  # ← AJOUT
        device=args.device,
        tensorboard_dir=tb_dir,
        train_scenarios=train_scenarios
    )
    
    print(f"Network: {sum(p.numel() for p in trainer.q_network.parameters()):,} parameters")
    
    # Resume
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        trainer.q_network.load_state_dict(checkpoint)
        print(f"Resumed from {args.resume_from}")
    
    # Train
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_freq']
    save_freq = config['training']['save_freq']
    
    print(f"\nStarting training for {num_episodes} episodes...")
    
    trainer.train(
        num_episodes=num_episodes,
        eval_freq=eval_freq,
        save_freq=save_freq,
        verbose=True
    )
    
    # Save
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    final_path = checkpoint_dir / 'final_model.pth'
    trainer.save_model(str(final_path))
    
    exp_name = config['logging']['experiment_name']
    output_path = f"src/submission/my_agent_dqn_{exp_name}.py"
    save_dqn_agent(trainer, output_path)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Submission: {output_path}")
    print(f"💾 Checkpoint: {final_path}")

if __name__ == '__main__':
    main()