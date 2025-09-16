import torch
import yaml
import gymnasium as gym
import sys
import numpy as np
import csv
from gymnasium.wrappers import RecordVideo

from Architecture import Actor

agent_file = sys.argv[1]
agent_params_file = sys.argv[2]

with open(agent_params_file, 'r') as f:
    hyperparams = yaml.safe_load(f)

env = gym.make(hyperparams['ENV_NAME'])
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = torch.tensor(env.action_space.high)

actor = Actor(obs_dim, action_dim, hyperparams['ACTOR_HIDDEN_LAYERS'], None, action_range)
actor.load_state_dict(torch.load(agent_file, weights_only=True))

rewards = []
actor.eval()
with torch.no_grad():
    for _ in range(hyperparams['NUM_TESTS']):
        state, _ = env.reset()
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

        rewards.append(0)
        while True:
            action = actor(state)
            next_state, reward, terminated, truncated, _ = env.step(np.asarray(action[0]))
            next_state = torch.tensor(next_state, dtype = torch.float32)

            rewards[-1] += reward
            state = next_state.unsqueeze(0)

            if terminated or truncated:
                break

with open(hyperparams['FILE_NAME'] + "TestingData.csv", 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(['run', 'score'])
    for run, score in enumerate(rewards):
        writer.writerow([run, score])

rewards = np.array(rewards)
summary = f"""
Total Runs: {hyperparams['NUM_TESTS']}
Mean Score: {np.mean(rewards):.2f}
Standard Deviation: {np.std(rewards):.2f}
Median Score: {np.median(rewards):.2f}
Min Score: {np.min(rewards):.2f}
Max Score: {np.max(rewards):.2f}
1st Quantile: {np.percentile(rewards, 25):.2f}
3rd Quantile: {np.percentile(rewards, 75):.2f}
"""

print(summary)

with open(hyperparams['FILE_NAME'] + "TestingDataSummary.txt", 'w', newline = '') as f:
    f.write(summary)

folder = hyperparams['FILE_NAME'].split('/')[1]

env.close()
env = gym.make(hyperparams['ENV_NAME'], render_mode='rgb_array')
env = RecordVideo(env, video_folder=folder, episode_trigger=lambda x: True)

state, _ = env.reset()
state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
with torch.no_grad():
    while True:
        action = actor(state)
        next_state, reward, terminated, truncated, _ = env.step(np.asarray(action[0]))
        next_state = torch.tensor(next_state, dtype = torch.float32)

        state = next_state.unsqueeze(0)

        if terminated or truncated:
            break

env.close()