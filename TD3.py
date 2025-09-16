import torch
from torch import optim, nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import yaml
import sys

from Architecture import *

hyperparams_file = sys.argv[1]
with open(hyperparams_file, 'r') as f:
    hyperparams = yaml.safe_load(f)

env = gym.make(hyperparams['ENV_NAME'])
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.shape[0]
action_range = torch.tensor(env.action_space.high)

actor, target_actor, critic, target_critic = get_architecture(OBS_SPACE, ACTION_SPACE, hyperparams['ACTOR_HIDDEN_LAYERS'], hyperparams['CRITIC_HIDDEN_LAYERS'], hyperparams['TAU'], action_range)
actor_optim = optim.Adam(actor.parameters(), float(hyperparams['ACTOR_LR']))
critic_optim = optim.Adam(critic.parameters(), float(hyperparams['CRITIC_LR']))

GAMMA = hyperparams['GAMMA']

NUM_EPOCHS = hyperparams['NUM_EPOCHS']
BATCH_SIZE = hyperparams['BATCH_SIZE']
TRAINING_START_STEP = hyperparams['TRAINING_START_STEP']

EXP_REPLAY_LENGTH= hyperparams['EXPERIENCE_REPLAY_LENGTH']

LOGGING_FREQ = hyperparams['LOGGINIG_FREQUECY']
SLIDING_WINDOW_SIZE = hyperparams['SLIDING_WINDOW_SIZE']

ACTOR_EXPLORATION_SIGMA = hyperparams['ACTOR_EXPLORATION_SIGMA']
CRITIC_SIGMA = hyperparams['CRITIC_SIGMA']

ACTOR_UPDATE_FREQ = hyperparams['ACTOR_UPDATE_FREQ']
cur_step = 0

rewards_over_time = []
actor_loss_array = []
critic_loss_array = []

mse = nn.MSELoss()
best_reward = -1000000

experience_replay = deque([], maxlen = EXP_REPLAY_LENGTH)

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

    current_reward = 0

    while True:
        with torch.no_grad():
            action = actor(state) + torch.normal(0,ACTOR_EXPLORATION_SIGMA, (1, ACTION_SPACE))
            action = torch.clamp(action, -action_range, action_range)
            next_state, reward, terminated, truncated, _ = env.step(np.asarray(action[0]))
            next_state = torch.tensor(next_state, dtype = torch.float32)
            experience_replay.append((state[0], action[0], reward, next_state, 1 if terminated else 0))
            current_reward += reward
            state = next_state.unsqueeze(0)
        
        if len(experience_replay) >= TRAINING_START_STEP:
            if len(experience_replay) == TRAINING_START_STEP:
                print(f"TRAINING STARTED ON EPOCH {epoch}, AVG REWARD {np.mean(rewards_over_time)}")

            random_experiences = random.sample(experience_replay, BATCH_SIZE)
            state_array, action_array, reward_array, next_state_array, done_array = zip(*random_experiences)

            state_array = torch.stack(state_array)
            action_array = torch.stack(action_array)
            reward_array = torch.tensor(reward_array, dtype = torch.float32).unsqueeze(1)
            next_state_array = torch.stack(next_state_array)
            done_array = torch.tensor(done_array, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                best_next_actions = target_actor(next_state_array)
                target_state_values = reward_array + GAMMA * torch.minimum(*target_critic(next_state_array, best_next_actions)) * (1 - done_array)
            predicted_state_values = critic(state_array, action_array)

            critic_optim.zero_grad()
            critic_loss = mse(target_state_values, predicted_state_values[0]) + mse(target_state_values, predicted_state_values[1])
            critic_loss.backward()
            critic_optim.step()

            if cur_step == 0:
                actor_optim.zero_grad()
                predicted_best_actions = actor(state_array) + torch.normal(0,CRITIC_SIGMA, (BATCH_SIZE, ACTION_SPACE))
                torch.clamp(predicted_best_actions, -action_range, action_range)

                actor_loss = -critic(state_array,predicted_best_actions)[0].mean()
                actor_loss.backward()
                actor_optim.step()

                target_actor.update(actor)
                target_critic.update(critic)

                actor_loss_array.append(actor_loss.detach().numpy())
            else:
                actor_loss_array.append(actor_loss_array[-1])
            critic_loss_array.append(critic_loss.detach().numpy())


            cur_step = (cur_step + 1) % ACTOR_UPDATE_FREQ

        if terminated or truncated:
            break

    rewards_over_time.append(current_reward)
    if epoch % LOGGING_FREQ == 0 and epoch != 0:
        print(f"Epoch {epoch}: Current Reward {rewards_over_time[-1]:.2f}, Avg Reward(Last {SLIDING_WINDOW_SIZE}) {np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):]):.2f}")
        if np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):]) > best_reward:
            best_reward = np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):])
            torch.save(actor.state_dict(), hyperparams['FILE_NAME'] + '.pt')

plt.plot(rewards_over_time, label = "Rewards", color = 'blue')
plt.plot([np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):epoch+1]) for epoch in range(NUM_EPOCHS)], label = 'Average Rewards', color='red')
plt.grid()
plt.legend()
plt.title("Rewards During Training")
plt.xlabel("Epoch")
plt.ylabel("Rewards")
plt.savefig(hyperparams['FILE_NAME'] + 'RewardsGraph.png')

plt.figure()
plt.plot(actor_loss_array, label = "Actor Loss", color = 'green')
plt.plot(critic_loss_array, label = "Critic Loss", color = 'blue')
plt.grid()
plt.legend()
plt.title("Loss During Training")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig(hyperparams['FILE_NAME'] + 'LossGraph.png')