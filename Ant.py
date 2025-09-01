import torch
from torch import nn,optim
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque

env = gym.make("Ant-v5")
ACTION_SPACE = env.action_space.shape[0]
OBS_SPACE = env.observation_space.shape[0]

GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4

EXPERIENCE_REPLAY_LENGTH = 300000

NUM_EPOCHS = 5000
BATCH_SIZE = 256
TRAINING_START_STEP = 10000

EXPLORATION_SIGMA = 0.1
SMOOTHING_SIGMA = 0.2
SMOOTHING_CLIP_BOUNDS = 0.5

ACTOR_UPDATE_PERIOD = 2
current_actor_step = 0

experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_LENGTH)

rewards_over_time = []

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_SPACE, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SPACE),
            nn.Tanh()
        )

    def forward(self, state):
        return self.actor(state)
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(ACTION_SPACE + OBS_SPACE, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

        self.critic2 = nn.Sequential(
            nn.Linear(ACTION_SPACE + OBS_SPACE, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, state, action):
        state_action_pair = torch.cat((state, action), dim=-1)
        return self.critic1(state_action_pair), self.critic2(state_action_pair)
    
actor = Actor()
actor_optim = optim.Adam(actor.parameters(), ACTOR_LR)
target_actor = Actor()
target_actor.load_state_dict(actor.state_dict())

critic = Critic()
critic_optim = optim.Adam(critic.parameters(), CRITIC_LR)
target_critic = Critic()
target_critic.load_state_dict(critic.state_dict())

mse = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32)
    reward_during_epoch = 0

    while True:
        with torch.no_grad():
            action = actor(state) + torch.normal(0,EXPLORATION_SIGMA, (ACTION_SPACE,))
            action = torch.clamp(action, -1,1)
            next_state, reward, terminated, truncated, _ = env.step(np.array(action))
            next_state = torch.tensor(next_state, dtype = torch.float32)
            experience_replay.append((state,action,reward,next_state, 1 if terminated else 0))
            state = next_state
            reward_during_epoch += reward

        if len(experience_replay) >= TRAINING_START_STEP:
            if len(experience_replay) == TRAINING_START_STEP:
                print(f"TRAINING STARTED ON EPOCH {epoch}: {np.mean(rewards_over_time):.2f}")
            random_sample = random.sample(experience_replay, BATCH_SIZE)
            state_array, action_array, reward_array, next_state_array, done_array = zip(*random_sample)

            state_array = torch.stack(state_array)
            action_array = torch.stack(action_array)
            reward_array = torch.tensor(reward_array, dtype = torch.float32)
            next_state_array = torch.stack(next_state_array)
            done_array = torch.tensor(done_array)

            with torch.no_grad():
                best_next_actions = target_actor(next_state_array)
                smoothing_noise = torch.clamp(torch.normal(0,SMOOTHING_SIGMA, size = (BATCH_SIZE, ACTION_SPACE)),-SMOOTHING_CLIP_BOUNDS, SMOOTHING_CLIP_BOUNDS)

                target_state_action_values = reward_array.unsqueeze(1) + GAMMA * torch.minimum(*target_critic(next_state_array, torch.clamp(best_next_actions + smoothing_noise,-1,1))) * (1 - done_array).unsqueeze(1)
            predicted_state_action_values1, predicted_state_action_values2 = critic(state_array, action_array)

            critic_loss = mse(target_state_action_values, predicted_state_action_values1) + mse(target_state_action_values, predicted_state_action_values2)

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optim.step()

            current_actor_step = (current_actor_step + 1) % ACTOR_UPDATE_PERIOD

            if current_actor_step == 0:
                best_actions = actor(state_array)
                actor_loss = -critic(state_array, best_actions)[0].mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                for params, target_params in zip(critic.parameters(), target_critic.parameters()):
                    target_params.data.copy_(TAU * params.data + (1 - TAU) * target_params.data)

                for params, target_params in zip(actor.parameters(), target_actor.parameters()):
                    target_params.data.copy_(TAU * params.data + (1 - TAU) * target_params.data)

        if terminated or truncated:
                break

    rewards_over_time.append(reward_during_epoch)

    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch}: Current Reward {rewards_over_time[-1]:.2f}, Avg Reward (Last 50) {np.mean(rewards_over_time[max(0, epoch - 50):]):.2f}")

torch.save(actor.state_dict(), "Ant.pt")

plt.plot(rewards_over_time, color = "blue", label = "Rewards")
plt.plot([np.mean(rewards_over_time[max(0, epoch - 50) : epoch]) for epoch in range(NUM_EPOCHS)], color = "red", label = "Average Rewards")
plt.grid()
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("Ant")
plt.show()