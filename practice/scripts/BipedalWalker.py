import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from collections import deque

from OUP import OUP

env = gym.make("Walker2d-v5")
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.shape[0]

oup = OUP(ACTION_SPACE, sigma = 0.5)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SPACE),
            nn.Tanh()
        )

    def forward(self, state):
        return self.actor(state)
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(OBS_SPACE + ACTION_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(OBS_SPACE + ACTION_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim = -1)
        return self.critic1(state_action), self.critic2(state_action)
def main():
    actor = Actor()
    actor_optimizer = optim.Adam(actor.parameters(), 3e-5)
    actor_target = Actor()
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic()
    critic_optimizer = optim.Adam(critic.parameters(), 3e-4)
    critic_target = Critic()
    critic_target.load_state_dict(critic.state_dict())

    mse = nn.MSELoss()

    GAMMA = 0.99
    TAU = 0.005
    NUM_EPOCHS = 4000
    BATCH_SIZE = 256
    TRAINING_START_STEP = 2000
    EXPERIENCE_REPLAY_LEN = 300000
    EXPLORATION_NOISE = 0.1
    SIGMA = 0.2
    D = 2
    current_step = 0
    CLIP_BOUNDS = 0.5

    experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_LEN)

    rewards_over_time = []

    for epoch in range(NUM_EPOCHS):
        state, _ = env.reset()
        oup.reset()

        state = torch.tensor(state, dtype = torch.float32)
        reward_during_epoch = 0

        while True:
            with torch.no_grad():
                action = actor(state) + torch.normal(0, EXPLORATION_NOISE, (ACTION_SPACE,))
                action = torch.clamp(action,-1,1)
                next_state, reward, terminated, truncated, _ = env.step(np.array(action))
                next_state = torch.tensor(next_state, dtype = torch.float32)
                experience_replay.append((state,action,reward,next_state, 1 if terminated else 0))
                reward_during_epoch += reward
                state = next_state
            if len(experience_replay) > TRAINING_START_STEP:
                with torch.no_grad():
                    random_sample = random.sample(experience_replay, BATCH_SIZE)
                    state_array, action_array, reward_array, next_state_array, done_array = zip(*random_sample)

                    state_array = torch.stack(state_array)
                    action_array = torch.stack(action_array)
                    reward_array = torch.tensor(reward_array, dtype = torch.float32)
                    next_state_array = torch.stack(next_state_array)
                    done_array = torch.tensor(done_array)

                    best_next_actions = actor_target(next_state_array)
                    epsilon = torch.clamp(torch.normal(0, SIGMA, size = (BATCH_SIZE, ACTION_SPACE)), -CLIP_BOUNDS, CLIP_BOUNDS)
                    critic1_guess, critic2_guess = critic_target(next_state_array, torch.clamp(best_next_actions + epsilon,-1,1))
                    target_state_action_values = reward_array.unsqueeze(1) + GAMMA * torch.min(critic1_guess, critic2_guess) * (1 - done_array).unsqueeze(1)
                predicted_state_action_values1, predicted_state_action_values2 = critic(state_array, action_array)

                critic_loss1 = mse(predicted_state_action_values1, target_state_action_values)
                critic_loss2 = mse(predicted_state_action_values2, target_state_action_values)
                
                critic_loss = critic_loss1 + critic_loss2

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()

                current_step += 1
                if current_step % D == 0:
                    new_actions = actor(state_array)
                    actor_loss = -critic(state_array,new_actions)[0].mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    new_actor_target_state_dict = {}
                    for key in actor.state_dict().keys():
                        new_actor_target_state_dict[key] = TAU * actor.state_dict()[key] + (1 - TAU) * actor_target.state_dict()[key]

                    actor_target.load_state_dict(new_actor_target_state_dict)

                    new_critic_target_state_dict = {}
                    for key in critic.state_dict().keys():
                        new_critic_target_state_dict[key] = TAU * critic.state_dict()[key] + (1 - TAU) * critic_target.state_dict()[key]

                    critic_target.load_state_dict(new_critic_target_state_dict)
            
            if terminated or truncated:
                break

        rewards_over_time.append(reward_during_epoch)

        if epoch % 100 == 0 and epoch != 0:
            print(f"Epoch {epoch}: Current Reward {rewards_over_time[-1]:.2f}, Average Reward (Last 50) {np.mean(rewards_over_time[max(0,epoch-50):]):.2f}")

    plt.plot(rewards_over_time, color = "blue", label = "Rewards")
    plt.plot([np.mean(rewards_over_time[max(0,epoch-50):epoch]) for epoch in range(NUM_EPOCHS)], color = "red", label = "Average Rewards")
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Bipedal Walker")
    plt.show()

    torch.save(actor.state_dict(), "BipedalWalker.pt")

if __name__ == '__main__':
    main()