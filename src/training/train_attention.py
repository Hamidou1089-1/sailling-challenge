#!/usr/bin/env python3
"""Training script for Attention-based DQN agent (NO REPLAY BUFFER)."""

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
    parser = argparse.ArgumentParser(description="Train Attention-based DQN agent (NO REPLAY BUFFER)")
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
    print(f"\n{'='*70}")
    print("ATTENTION DQN - NO REPLAY BUFFER VERSION")
    print(f"{'='*70}")
    print("Training strategy:")
    print("  1. Collect full episode in unique wind configuration")
    print("  2. Train multiple epochs on this episode")
    print("  3. Discard data and move to new wind configuration")
    print(f"{'='*70}\n")
    
    # Create directories
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path('src/submission').mkdir(parents=True, exist_ok=True)
    
    train_scenarios = config['training']['train_scenarios']
    print(f"Training scenarios: {train_scenarios}")
    
    # Create environment
    env = SailingEnv(**get_wind_scenario(train_scenarios[0]))
    
    exp_name = config['logging']['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = f"runs/attention_no_replay/{exp_name}_{timestamp}"
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    print(f"📊 TensorBoard dir: {tb_dir}")
    
    # Create trainer
    print("\n🎮 Creating Attention DQN trainer (NO REPLAY BUFFER)...")
    trainer = DQNTrainer(
        env,
        # Learning
        learning_rate=config['agent']['learning_rate'],
        lr_decay=config['agent'].get('lr_decay', 0.9999),
        gamma=config['agent']['gamma'],
        
        # Training (NO BUFFER)
        # batch_size=config['agent']['batch_size'],
        # n_epoch_per_episode=config['agent'].get('n_epoch_per_episode', 4),
        
        # Target
        target_update_freq=config['agent']['target_update_freq'],
        
        # Algorithm
        use_double_dqn=config['agent'].get('use_double_dqn', True),
        use_noisy_net=config['agent'].get('use_noisy_net', True),
        
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
    
    print(f"\n📊 Network Details:")
    print(f"   Total parameters: {sum(p.numel() for p in trainer.q_network.parameters()):,}")
    print(f"   Architecture:")
    print(f"     - d_model: {config['agent'].get('d_model', 64)}")
    print(f"     - n_heads: {config['agent'].get('n_heads', 4)}")
    print(f"     - n_layers: {config['agent'].get('n_layers', 2)}")
    print(f"     - n_queries: {config['agent'].get('n_queries', 4)}")
    print(f"   Algorithm:")
    print(f"     - NoisyNet: {trainer.use_noisy_net}")
    print(f"     - Double DQN: {trainer.use_double_dqn}")
    print(f"   Training:")
    print(f"     - Batch size: {config['agent']['batch_size']}")
    print(f"     - Epochs per episode: {config['agent'].get('n_epoch_per_episode', 4)}")
    
    # Resume
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        trainer.q_network.load_state_dict(checkpoint)
        trainer.target_network.load_state_dict(checkpoint)
        print(f"\n✅ Resumed from {args.resume_from}")
    
    # Train
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_freq']
    save_freq = config['training']['save_freq']
    
    print(f"\n🚀 Starting training for {num_episodes} episodes...")
    print(f"   Eval frequency: every {eval_freq} episodes")
    print(f"   Save frequency: every {save_freq} episodes")
    print(f"\n{'='*70}\n")
    
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
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"📁 Submission: {output_path}")
    print(f"💾 Checkpoint: {final_path}")
    print(f"\n📊 View training progress:")
    print(f"   tensorboard --logdir={tb_dir}")
    print(f"\n🎯 Training strategy used:")
    print(f"   - NO REPLAY BUFFER")
    print(f"   - Each episode trained {config['agent'].get('n_epoch_per_episode', 4)} times")
    print(f"   - {num_episodes} unique wind configurations experienced")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()