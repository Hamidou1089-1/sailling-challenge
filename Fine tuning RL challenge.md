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

# DQN Baseline Configuration
# Safe conservative settings for initial training

# Training parameters
training:
  num_episodes: 80000
  max_steps_per_episode: 200
  eval_freq: 100          # Evaluate every N episodes
  save_freq: 5000        # Save checkpoint every N episodes
  log_freq: 1000            # Log to TensorBoard every N episodes
  
  # Wind scenarios for training
  train_scenarios: ['training_1', 'training_2', 'training_3']
  eval_scenarios: ['training_1', 'training_2', 'training_3']
  
  # Reproducibility
  seed: 42

# Agent hyperparameters
agent:
  # Learning rate
  learning_rate: 0.004   # Conservative 3e-4
  lr_decay: 0.999998        # Slow decay
  min_lr: 0.00012         # Floor
  
  # Exploration (epsilon-greedy)
  epsilon_start: 1.0
  epsilon_end: 0.0002       # Changed from 0.05 to 0.01 for more exploitation
  epsilon_decay: 0.999998   # Linear decay over ~14k steps to reach 0.01
  
  # Discount factor
  gamma: 0.99
  
  # Experience replay
  buffer_capacity: 200000
  batch_size: 64
  learning_starts: 500   # Start learning after N steps
  
  # Target network
  target_update_type: "hard"  # Options: "hard" or "soft"
  target_update_freq: 5000    # For hard update (every N steps)
  tau: 0.005                  # For soft update (not used if hard)
  
  # Network architecture
  use_double_dqn: true        # Enable Double DQN (recommended)
  gradient_clip: 10.0         # Clip gradients
  
# Environment
environment:
  goal: [16, 31]
  grid_size: [32, 32]


# Checkpointing
checkpoint:
  save_dir: "checkpoints/dqn"
  resume_from: null           # Path to checkpoint to resume from

# Logging
logging:
  tensorboard_dir: "runs/dqn"
  experiment_name: "dqn_baseline"
  log_gradients: false        # Log gradient histograms (expensive)

resultat:

Best Agent on codabench.

Reports structure : 
- Lets define the environnement, the number of possible states.
- Why the challenge isn't trivial
- what kind of different approach we can have to compete for a better score
- What is my approach, physics shaping, reward shapping, why i didn't choose n step, ucb approach, balance between complexity and compute power
- Where did I fail
- Why did I choose DQN
- Noisy + PER
- My next step
- If more compute power (which I had access to) what will be the next step
- Parameters to reproduce my results

The number of possible states of the environnement is infinite (because of the continuous variables).
the number of possible action is 9, for all the direction in space discritize (because we can have for the same environnement, a continuous (angle) action space).

One possible answer for the non triviality of the challenge is the number of possible states is infinite, so it's computational impossible to compute a q table
that can helps us decide for the action we take for the sailling.
And one other answer is that even if we discritize (will be define afterward), we have to make a trade off between number of bins to discritize the continuous 
variable and not making the q table to big or else it will also be computationnal challenging.

Ok so in RL, we have some algorithme which have more potential to succeed for certain task:
- We have two cases possible :
  - We discretize the observation, in that case we can apply 
Here we have a continous space, if don't discretize it, the classical q learning algorithme is disqualified.
But algorithme with ucb did well on the training data 50 on average reward and 71 in steps, but on the hidden test data, i did very poorly, 2.29 without ucb,
and 9 with ucb, that wasn't the fault of the algorithme itself, my training strategie was quite bad, training without any validation step to finitune is useless in ml in general.
so i did some measurement during my fine tune phase, which was essently tuning the reward and the physics feature.
My goal was to help my agent with good information, and good feedback, because in this environnement, we can with infinite compute power we can make an almost 
perfect planning for the agent (rule based approach), but the rl way is try and fail and learn if possible.
so the trade off was to give him enough dopamine for going fast and for getting closer, but we don't want him to be a junky, so the reward he gets from the immadiate feedback shouldn't replace the final reward and the number of step, so the goal of my finetuning was to handle that.
at some point i named the coefficient for the speed of the boat mu, and prog for the progression made, and took -1 at each step to keep the agent for not mooving when everything in the environnement is against him (wind speed etc).
and thus my implementation of dqn came from my faillure to tune well enough my classical algorithm, but to be honnest i didn't fully commit to it because from the start i knew that they were limited because of the discretize approach which made the information given to the agent a very poor version of the real state of the 
environnement.
So i went to geeks for geeks, and medium to see different algorithme for continuous states, what are there advantages etc, and dqn was the simplest and maybe 
the most efficient to implement for a first try.
and it didn't dissapoint, for my first complete trained model, I beat the rule based model on codabench and took the first place.
the architecture of the model evolved so my first submissions isn't my first dqn model, the first one didn't had cnn and average pooling to it.
I decide to add a cnn with average pooling because of the information approach and efficiency, first we capture the whole map, and then we reduce until we only see the local information, the goal for this construction was to help for better planning, and i hoped that the model could predict the wind 6 position or more can influence in maybe 3 steps my decision to maximize my speed toward the goal.
and then the rest of the architecture was prety classique, it was the implementation from medium that i was following.
I also wanted to have a better exploration strategy as in classical q learning with ucb, but turnout in dqn it's quite difficult, because i choosed noisy net for 
the natural exploration approach by only adding stochasticity to the weigh of the model, it did better than the rule based model, but less than my classical dqn model, and i even added PER, prioritized experience replay, which was suppose to help me choose with some priority which episode to replay, but turnout when doing a lot of episode, and with my randomisation environnement approach, it just give to much importance to outliers in my buffer, but still being ok, not improving.
my next step was to implement PPO, because i notice that there is some lack of robustness to my off policy approach, and also because i wanted to try to see 
how it will behave, it simple to implement, and not more difficult to fine tune than dqn, so there it is.

