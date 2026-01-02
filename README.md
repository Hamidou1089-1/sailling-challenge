# Sailing RL Challenge 🚤

Deep Reinforcement Learning project for navigating a sailboat under varying wind conditions.

![Sailing Challenge](illustration_challenge.png)

## 📋 Project Structure

```
sailing-rl-challenge/
├── config/                      # Configuration files (YAML)
│   ├── dqn_baseline.yaml       # DQN hyperparameters
│   └── qlearning_baseline.yaml # Q-Learning hyperparameters
│
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py       # [PROF] Base interface
│   │   ├── my_agent.py         # Q-Learning agent
│   │   └── my_agent_DQN.py     # DQN agent + trainer
│   │
│   ├── networks/               # Neural networks
│   │   └── q_network.py        # CNN + MLP architecture
│   │
│   ├── utils/                  # Utility functions
│   │   ├── agent_utils.py      # save_my_agent()
│   │   └── save_my_dqn.py      # save_dqn_agent()
│   │
│   ├── training/               # Training scripts
│   │   ├── train_dqn.py        # DQN training script
│   │   └── train_qlearning.py  # Q-Learning training script
│   │
│   └── submission/             # Generated submission files
│
├── wind_scenarios/             # [PROF] Environment files (immuable)
│   ├── __init__.py
│   ├── env_sailing.py
│   ├── sailing_physics.py
│   ├── evaluation.py
│   ├── evaluate_submission.py
│   └── test_agent_validity.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── train_colab.ipynb       # Training on Colab with GPU
│   └── analyze_results.ipynb   # Results analysis
│
├── checkpoints/                # Model checkpoints (gitignored)
├── runs/                       # TensorBoard logs (gitignored)
├── submissions/                # Final submission files
├── data/                       # Normalization stats, etc.
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sailing-challenge.git
cd sailing-challenge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install in development mode
pip install -e .
```

### Training Locally

**Q-Learning (CPU only):**

```bash
python src/training/train_qlearning.py --config config/qlearning_baseline.yaml
```

**DQN (CPU or GPU):**

```bash
# 1. Collect normalization statistics (once)
python src/training/train_dqn.py --collect-stats --config config/dqn_baseline.yaml

# 2. Train the agent
python src/training/train_dqn.py --config config/dqn_baseline.yaml
```

### Training on Google Colab

1. Upload your repository to GitHub
2. Open `notebooks/train_colab.ipynb` in Colab
3. Mount Google Drive for checkpoints
4. Run all cells

### Evaluation

```bash
# Validate your agent
python wind_scenarios/test_agent_validity.py src/submission/my_agent.py

# Evaluate performance
python wind_scenarios/evaluate_submission.py src/submission/my_agent.py \
    --seeds 1 --num-seeds 100 --verbose
```

## 🎯 Challenge Overview

Train an agent to navigate a sailboat from start to goal under varying wind conditions. The challenge includes:

- **3 training scenarios** with different wind patterns
- **1 hidden test scenario** for final evaluation
- Realistic sailing physics (wind angle, momentum, no-go zones)
- Strategic planning required (can't sail directly into wind)

**Goal:** Learn a generalizable policy that performs well on unseen wind conditions.

## 📊 Configuration

Hyperparameters are managed via YAML files in `config/`. Key parameters:

### DQN (`config/dqn_baseline.yaml`)

- Learning rate: 3e-4 with slow decay
- Epsilon: 1.0 → 0.01 over training
- Buffer size: 50k transitions
- Target update: Hard update every 1000 steps
- Double DQN: Enabled

### Q-Learning (`config/qlearning_baseline.yaml`)

- Learning rate: 0.1 → 0.005 with decay
- Exploration: UCB-based (no epsilon)
- Reward shaping: Progress (μ=9.0) + Velocity (ν=0.39)
- State space: 12 distance bins × 12 angle bins + wind features

## 🔧 Development Workflow

1. **Local development:** Test ideas quickly on CPU
2. **Push to GitHub:** Version control your experiments
3. **Train on Colab:** Use free GPU for DQN training
4. **Download checkpoints:** Save to Google Drive
5. **Generate submission:** `save_dqn_agent()` creates standalone file
6. **Validate & Submit:** Test then upload to Codabench

## 📈 Monitoring

TensorBoard logs are saved to `runs/`:

```bash
tensorboard --logdir runs/
```

Metrics logged:

- Episode reward and success rate
- Q-value statistics
- Loss curves
- Epsilon decay
- Learning rate schedule

## 🧪 Experiments

Create new configs for experiments:

```bash
cp config/dqn_baseline.yaml config/dqn_experiment1.yaml
# Edit hyperparameters
python src/training/train_dqn.py --config config/dqn_experiment1.yaml
```

## 📝 Submission Format

Your agent must:

1. Inherit from `BaseAgent`
2. Implement `act(observation) -> int`
3. Be completely self-contained (no external files)
4. Import: `from evaluator.base_agent import BaseAgent` (for Codabench)

The utility functions `save_my_agent()` and `save_dqn_agent()` handle this automatically.

## 🏆 Best Practices

- **Start simple:** Train Q-Learning first to understand the environment
- **Visualize:** Use notebooks to inspect trajectories
- **Monitor:** Check TensorBoard regularly for instabilities
- **Checkpoint often:** Save every 1000 episodes
- **Test generalization:** Evaluate on all 3 training scenarios
- **Avoid overfitting:** Don't tune on the hidden test scenario!

## 🤝 Contributing

This is a course project, but improvements to the infrastructure are welcome:

- Better logging/visualization
- Hyperparameter optimization tools
- Additional RL algorithms

## 📧 Contact

For questions about the challenge: t.rahier@criteo.com

## 📄 License

MIT License - See LICENSE file for details

---

**Note:** Files in `wind_scenarios/` are provided by the professor and should NOT be modified, as they define the official evaluation environment.
