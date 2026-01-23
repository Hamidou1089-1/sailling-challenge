#!/usr/bin/env python3
"""Training script for Attention-based DQN agent."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.my_agent_attention import DQNTrainer
from src.utils.save_attention_based import save_attention_agent
from wind_scenarios import get_wind_scenario
from wind_scenarios.env_sailing import SailingEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention-based DQN agent")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
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
    
    train_scenarios = config['training']['train_scenarios']
    print(f"Training scenarios: {train_scenarios}")
    
    # Create environment
    env = SailingEnv(**get_wind_scenario(train_scenarios[0]))
    
    exp_name = config['logging']['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = f"runs/attention/{exp_name}_{timestamp}"
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    print(f"📊 TensorBoard dir: {tb_dir}")
    
    # Create trainer
    print("\n🎮 Creating Attention DQN trainer...")
    trainer = DQNTrainer(
        env,
        # Learning
        learning_rate=config['agent']['learning_rate'],
        lr_decay=config['agent'].get('lr_decay', 0.9999),
        gamma=config['agent']['gamma'],
        
        # Buffer
        buffer_capacity=config['agent']['buffer_capacity'],
        batch_size=config['agent']['batch_size'],
        
        # Target
        target_update_freq=config['agent']['target_update_freq'],
        
        # Algorithm
        use_double_dqn=config['agent'].get('use_double_dqn', True),
        use_noisy_net=config['agent'].get('use_noisy_net', True),
        use_per=config['agent'].get('use_per', True),
        per_alpha=config['agent'].get('per_alpha', 0.6),
        per_beta_start=config['agent'].get('per_beta_start', 0.4),
        per_beta_frames=config['agent'].get('per_beta_frames', 120000),
        
        # Attention architecture
        d_model=config['agent'].get('d_model', 64),
        n_heads=config['agent'].get('n_heads', 4),
        n_layers=config['agent'].get('n_layers', 2),
        n_queries=config['agent'].get('n_queries', 4),
        patch_size=config['agent'].get('patch_size', 4),
        dropout=config['agent'].get('dropout', 0.1),
        
        device=args.device,
        tensorboard_dir=tb_dir,
        train_scenarios=train_scenarios
    )
    
    print(f"\nNetwork: {sum(p.numel() for p in trainer.q_network.parameters()):,} parameters")
    print(f"   NoisyNet: {trainer.use_noisy_net}")
    print(f"   PER: {trainer.use_per}")
    print(f"   Double DQN: {trainer.use_double_dqn}")
    
    # Resume
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        trainer.q_network.load_state_dict(checkpoint)
        trainer.target_network.load_state_dict(checkpoint)
        print(f"✓ Resumed from {args.resume_from}")
    
    # Train
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_freq']
    save_freq = config['training']['save_freq']
    
    print(f"\n🚀 Starting training for {num_episodes} episodes...")
    print(f"   Eval frequency: every {eval_freq} episodes")
    print(f"   Save frequency: every {save_freq} episodes")
    
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
    output_path = f"src/submission/my_agent_{exp_name}.py"
    save_attention_agent(trainer, output_path)
    
    print(f"\n✅ Training complete!")
    print(f"📝 Submission: {output_path}")
    print(f"💾 Checkpoint: {final_path}")
    print(f"\n📊 View training progress:")
    print(f"   tensorboard --logdir={tb_dir}")

if __name__ == '__main__':
    main()