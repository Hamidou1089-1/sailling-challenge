---
noteId: "f3a40530e64511f08e50a5f8acb448a2"
tags: []
---
> # Fine tuning RL challenge

> ## *Q Learning TD(0)*:

**Parameters:**

- Learning rate $\alpha_t$
- Exploration rate $\gamma_t$
- progress reward coeff $\mu_t$
- velocity reward coeff $\nu_t$

Test 1 : Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.998)
- $\gamma_t$ = decay : max(0.05, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.19
  Res :


| WIND_SCENARIO |   SUCCESS RATE   |      MEAN REWARD      |           MEAN STEPS |
| :------------ | :--------------: | :--------------------: | -------------------: |
| training_1    | Success: 100.00% | Reward: 26.52 ± 4.56 | Steps: 134.4 ± 16.4 |
| training_2    | Success: 100.00% | Reward: 54.34 ± 23.95 |  Steps: 76.5 ± 60.0 |
| training_3    | Success: 100.00% | Reward: 43.00 ± 8.15 |  Steps: 89.9 ± 41.8 |

OVERALL      | Success: 100.00% ± 0.00%
Reward: 41.29 ± 11.42
Steps: 100.3 ± 24.8

Test 2 : Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.99)
- $\gamma_t$ = decay : max(0.05, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.09
  Res :


  | WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
  | ------------- | ---------------- | ---------------------- | -------------------- |
  | training_1    | Success: 100.00% | Reward: 26.99 ± 5.82  | Steps: 133.8 ± 22.5 |
  | training_2    | Success: 100.00% | Reward: 35.30 ± 23.28 | Steps: 131.8 ± 78.5 |
  | training_3    | Success: 100.00% | Reward: 43.07 ± 5.72  | Steps: 85.8 ± 15.1  |

> OVERALL
>
> - Success: 100.00% ± 0.00%
> - Reward: 35.12 ± 6.57
> - Steps: 117.2 ± 22.1

**Test 2:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.001, ql_agent_full.learning_rate \* 0.998)
- $\gamma_t$ = decay : max(0.05, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.09

Res :


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
| ------------- | ---------------- | ---------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 27.05 ± 4.75  | Steps: 132.6 ± 17.1 |
| training_2    | Success: 100.00% | Reward: 49.99 ± 25.20 | Steps: 89.4 ± 69.6  |
| training_3    | Success: 100.00% | Reward: 41.35 ± 7.69  | Steps: 92.2 ± 33.7  |

> OVERALL:
>
> - Success: 100.00% ± 0.00%
> - Reward: 39.46 ± 9.46
> - Steps: 104.7 ± 19.7

**Test 3:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : 0.1 / (1 + episode / 1000)
- $\gamma_t$ = decay : max(0.05, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.09

RES:


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
| ------------- | ---------------- | ---------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 27.11 ± 4.29  | Steps: 132.1 ± 15.6 |
| training_2    | Success: 100.00% | Reward: 48.94 ± 25.52 | Steps: 92.9 ± 72.1  |
| training_3    | Success: 100.00% | Reward: 39.79 ± 7.19  | Steps: 94.5 ± 19.2  |

> OVERALL:
>
> - Success: 100.00% ± 0.00%
> - Reward: 38.61 ± 8.95
> - Steps: 106.5 ± 18.1

**Test 4:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : 0.1 / (1 + episode / 1000)
- $\gamma_t$ = decay : max(0.005, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.09

**Test 5:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.998)
- $\gamma_t$ = decay : max(0.05, ql_agent_full.exploration_rate \* 0.98)
- $\mu_t$ = 9
- $\nu_t$ = 0.29

RES:


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
| ------------- | ---------------- | ---------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 21.16 ± 5.03  | Steps: 158.0 ± 21.9 |
| training_2    | Success: 100.00% | Reward: 60.90 ± 22.17 | Steps: 63.0 ± 59.6  |
| training_3    | Success: 100.00% | Reward: 42.28 ± 5.39  | Steps: 87.5 ± 12.9  |

> OVERALL:
>
> - Success: 100.00% ± 0.00%
> - Reward: 41.45 ± 16.24
> - Steps: 102.8 ± 40.3

## Q learning with UCB

**Parameters:**

- Learning rate $\alpha_t$
- Exploration rate $\gamma_t$ : *No longer needed, because ucb*
- progress reward coeff $\mu_t$
- velocity reward coeff $\nu_t$

**Test 1:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.998)
- $\mu_t$ = 9
- $\nu_t$ = 0.29


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
| ------------- | ---------------- | ---------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 25.17 ± 3.47  | Steps: 139.2 ± 13.9 |
| training_2    | Success: 100.00% | Reward: 72.70 ± 3.53  | Steps: 32.8 ± 4.8   |
| training_3    | Success: 100.00% | Reward: 37.05 ± 12.61 | Steps: 115.0 ± 71.6 |

> OVERALL:
>
> - Success: 100.00% ± 0.00%
> - Reward: 44.97 ± 20.20
> - Steps: 95.7 ± 45.5

**Test 2:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.998)
- $\mu_t$ = 9
- $\nu_t$ = 0.39


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD            | MEAN STEPS           |
| ------------- | ---------------- | ---------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 31.24 ± 8.71  | Steps: 120.5 ± 27.3 |
| training_2    | Success: 100.00% | Reward: 71.93 ± 6.98  | Steps: 34.5 ± 13.2  |
| training_3    | Success: 100.00% | Reward: 36.79 ± 10.52 | Steps: 109.7 ± 53.7 |

OVERALL:

- Success: 100.00% ± 0.00%
- Reward: 46.65 ± 18.02
- Steps: 88.2 ± 38.3

## Q learning with UCB and explicit penalize step

**Parameters:**

- Learning rate $\alpha_t$
- Exploration rate $\gamma_t$ : *No longer needed, because ucb*
- progress reward coeff $\mu_t$
- velocity reward coeff $\nu_t$

**Test 1:** Training on 1, with 1000 episodes, max step = 200

- $\alpha_t$ = decay : max(0.005, ql_agent_full.learning_rate \* 0.998)
- $\mu_t$ = 9
- $\nu_t$ = 0.39

Res : false, need some weird finetuning (training with directness, but infering without it)


| WIND_SCENARIO | SUCCESS RATE     | MEAN REWARD           | MEAN STEPS           |
| ------------- | ---------------- | --------------------- | -------------------- |
| training_1    | Success: 100.00% | Reward: 32.00 ± 7.66 | Steps: 117.0 ± 22.9 |
| training_2    | Success: 100.00% | Reward: 72.53 ± 3.69 | Steps: 33.1 ± 5.0   |
| training_3    | Success: 100.00% | Reward: 47.82 ± 3.78 | Steps: 74.7 ± 8.0   |

OVERALL

- Success: 100.00% ± 0.00%
- Reward: 50.78 ± 16.68
- Steps: 74.9 ± 34.3




# DQN + CNN + Average pooling


configuration file (yaml)
\# DQN Baseline Configuration
\# Safe conservative settings for initial training

\# Training parameters
training:
  num_episodes: 6000
  max_steps_per_episode: 200
  eval_freq: 200          # Evaluate every N episodes
  save_freq: 200        # Save checkpoint every N episodes
  log_freq: 10            # Log to TensorBoard every N episodes
  
  \# Wind scenarios for training
  train_scenarios: ['training_1', 'training_2', 'training_3']
  eval_scenarios: ['training_1', 'training_2', 'training_3']
  
  \# Reproducibility
  seed: 42

\# Agent hyperparameters
agent:
  \# Learning rate
  learning_rate: 0.001   # Conservative 3e-4
  lr_decay: 0.99998        # Slow decay
  min_lr: 0.0003         # Floor
  
  \# Exploration (epsilon-greedy)
  epsilon_start: 1.0
  epsilon_end: 0.005       # Changed from 0.05 to 0.01 for more exploitation
  epsilon_decay: 0.9998   # Linear decay over ~14k steps to reach 0.01
  
  \# Discount factor
  gamma: 0.99
  
  \# Experience replay
  buffer_capacity: 100000
  batch_size: 64
  learning_starts: 200   # Start learning after N steps
  
  \# Target network
  target_update_type: "hard"  # Options: "hard" or "soft"
  target_update_freq: 500    # For hard update (every N steps)
  tau: 0.005                  # For soft update (not used if hard)
  
  \# Network architecture
  use_double_dqn: true        # Enable Double DQN (recommended)
  gradient_clip: 10.0         # Clip gradients
  
\# Environment
environment:
  goal: [16, 31]
  grid_size: [32, 32]

\# Normalization stats
normalization:
  stats_path: "data/normalization_stats.pkl"
  collect_n_episodes: 1500    # Episodes to collect stats

\# Checkpointing
checkpoint:
  save_dir: "checkpoints/dqn"
  resume_from: null           # Path to checkpoint to resume from

\# Logging
logging:
  tensorboard_dir: "runs/dqn"
  experiment_name: "dqn_baseline"
  log_gradients: false        # Log gradient histograms (expensive)


resultat:


sailling-challenge git:main*  196s
(.venv) ❯ python3 wind_scenarios/evaluate_submission.py src/submission/my_agent_dqn_dqn_baseline.py --seeds 1 --num-seeds 150
Loaded agent: MyAgentDQN

Evaluating on 3 wind scenarios with 150 seeds
Agent: MyAgentDQN
Maximum steps per episode: 1000

| WIND_SCENARIO    | SUCCESS RATE | MEAN REWARD       | MEAN STEPS |
| ------------ | ------------ | ------------ | ------------ |
| training_1   | Success: 100.00% | Reward: 60.98 ± 5.27 | Steps: 50.6 ± 8.9 |
| training_2   | Success: 100.00% | Reward: 76.51 ± 1.20 | Steps: 27.7 ± 1.6 |
| training_3   | Success: 100.00% | Reward: 61.85 ± 5.64 | Steps: 49.3 ± 10.0 |
OVERALL      | Success: 100.00% ± 0.00%
Reward: 66.45 ± 7.12
Steps: 42.5 ± 10.5