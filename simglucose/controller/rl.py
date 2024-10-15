import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Specify the environment
env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# Ornstein-Uhlenbeck Noise
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Experience Replay Buffer
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch

# Actor Model
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) * upper_bound
        return x

# Critic Model
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1_state = nn.Linear(num_states, 16)
        self.fc2_state = nn.Linear(16, 32)
        self.fc1_action = nn.Linear(num_actions, 32)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.fc1_state(state))
        state_out = torch.relu(self.fc2_state(state))

        action_out = torch.relu(self.fc1_action(action))

        concat = torch.cat([state_out, action_out], dim=1)
        x = torch.relu(self.fc3(concat))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

# Target Network Update Function
def update_target(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

# Noise generator
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Initialize actor and critic models
actor_model = Actor()
critic_model = Critic()

target_actor = Actor()
target_critic = Critic()

# Copy weights from actor/critic to target networks
target_actor.load_state_dict(actor_model.state_dict())
target_critic.load_state_dict(critic_model.state_dict())

# Optimizers
critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.002)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)

# Hyperparameters
total_episodes = 100
gamma = 0.99
tau = 0.005
buffer = Buffer(50000, 64)

# Policy function
def policy(state, noise_object):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    sampled_actions = actor_model(state).detach().numpy()
    noise = noise_object()
    sampled_actions = sampled_actions + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.squeeze(legal_action)

# Training loop
ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        action = policy(prev_state, ou_noise)
        state, reward, done, truncated, _ = env.step(action)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        if buffer.buffer_counter >= buffer.batch_size:
            state_batch, action_batch, reward_batch, next_state_batch = buffer.sample()

            with torch.no_grad():
                target_actions = target_actor(next_state_batch)
                y = reward_batch + gamma * target_critic(next_state_batch, target_actions)

            critic_value = critic_model(state_batch, action_batch)
            critic_loss = torch.mean((y - critic_value) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actions = actor_model(state_batch)
            actor_loss = -torch.mean(critic_model(state_batch, actions))

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            update_target(target_actor, actor_model, tau)
            update_target(target_critic, critic_model, tau)

        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting results
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
