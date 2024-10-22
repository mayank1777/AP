import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from .base import Action
from .base import Controller

# Ornstein-Uhlenbeck Noise
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std_dev = torch.tensor(std_deviation, dtype=torch.float32)
        self.dt = dt
        self.x_initial = torch.tensor(x_initial, dtype=torch.float32) if x_initial is not None else None
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(self.mean))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else torch.zeros_like(self.mean)

# Experience Replay Buffer
class Buffer:
    def __init__(self, statedim, actiondim, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, statedim))
        self.action_buffer = np.zeros((self.buffer_capacity, actiondim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, statedim))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]  # Include done
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)
        done_batch = torch.tensor(self.done_buffer[batch_indices], dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

# Actor Model
class Actor(nn.Module):
    def __init__(self, statedim, actiondim, upper_bound):
        super(Actor, self).__init__()
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32)
        self.fc1 = nn.Linear(statedim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, actiondim)  # Output kp, ki, kd
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) * self.upper_bound
        return x

# Critic Model
class Critic(nn.Module):
    def __init__(self, statedim, actiondim):
        super(Critic, self).__init__()
        self.fc1_state = nn.Linear(statedim, 16)
        self.fc2_state = nn.Linear(16, 32)
        self.fc1_action = nn.Linear(actiondim, 32)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.fc1_state(state))
        state_out = torch.relu(self.fc2_state(state_out))

        action_out = torch.relu(self.fc1_action(action))

        concat = torch.cat([state_out, action_out], dim=1)
        x = torch.relu(self.fc3(concat))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class RLPIDController(Controller):
    def __init__(self, kp=1, ki=0, kd=0, target=90, k=5, action_dim=3, buffer_capacity=50000, batch_size=64, gamma=0.99, tau=0.005):
        """
        Initialize RLPIDController class, which uses DDPG to learn the PID controller gains.
        """
        self.kp = kp
        self.ki = kp
        self.kd = kp

        # Define upper and lower bounds as PyTorch tensors
        self.upper_bound = torch.tensor([10.0, 5.0, 1.0], dtype=torch.float32)
        self.lower_bound = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        self.target = target
        self.k = k  # Number of previous values to track
        self.state_dim = 2 * k  # State now contains k previous values of bg and meal
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize actor and critic models
        self.actor_model = Actor(self.state_dim, action_dim, self.upper_bound)
        self.critic_model = Critic(self.state_dim, action_dim)
        self.target_actor = Actor(self.state_dim, action_dim, self.upper_bound)
        self.target_critic = Critic(self.state_dim, action_dim)

        # Copy weights from actor/critic to target networks
        self.target_actor.load_state_dict(self.actor_model.state_dict())
        self.target_critic.load_state_dict(self.critic_model.state_dict())

        # Optimizers
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=0.002)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0.001)

        # Replay buffer
        self.buffer = Buffer(self.state_dim, action_dim, buffer_capacity, batch_size)

        # Exploration noise
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=float(self.std_dev) * np.ones(action_dim))

        # Training parameters
        self.ep_reward_list = []  # To store rewards for each episode
        self.avg_reward_list = []  # To store average rewards over episodes

        # Internal PID states
        self.integrated_state = 0
        self.bg_history = [0.0] * k  # History of previous k bg values
        self.meal_history = [0.0] * k  # History of previous k meal values

    def policy(self, observation, reward, done, **kwargs):
        """
        Policy function to compute PID control input with delta changes from the actor model and added exploration noise.
        """
        sample_time = kwargs.get('sample_time', 1)  # Default sample time to 1 second
        bg = observation.CGM  # Current observation (e.g., blood glucose level)
        meal = kwargs.get('meal')  # unit: g/min

        # Update the history lists
        self.bg_history.pop(0)
        self.bg_history.append(bg)
        self.meal_history.pop(0)
        self.meal_history.append(meal)

        # Construct state representation for the RL agent
        state = torch.tensor(self.bg_history + self.meal_history, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Use the actor model to get the deltas for PID gains (Δkp, Δki, Δkd)
        delta_pid_gains = self.actor_model(state).detach()  # Keep as tensor

        # Add exploration noise to the deltas (encourage exploration during training)
        noise = torch.tensor(self.ou_noise(), dtype=torch.float32)
        delta_pid_gains = delta_pid_gains + noise

        # Clip the deltas within reasonable bounds using torch.clamp()
        delta_pid_gains = torch.clamp(delta_pid_gains, self.lower_bound, self.upper_bound)

        # Convert back to NumPy for use in calculations (if needed)
        delta_kp, delta_ki, delta_kd = delta_pid_gains.squeeze().numpy()

        # Update the PID gains with the deltas
        self.kp += delta_kp
        self.ki += delta_ki
        self.kd += delta_kd

        control_input = (self.kp * (bg - self.target) +
                         self.ki * self.integrated_state +
                         self.kd * (bg - self.bg_history[-2]) / sample_time)

        # Update internal PID states
        self.integrated_state += (bg - self.target) * sample_time

        # Return the control action (e.g., basal insulin dose)
        # If control_input is a tensor, extract its scalar value
        basal = control_input.item() if isinstance(control_input, torch.Tensor) else control_input

        # Return Action with basal as a scalar and delta_kp, delta_ki, delta_kd
        return Action(basal=basal, bolus=0), delta_kp, delta_ki, delta_kd

    def step_and_record(self, observation, action, reward, done, **kwargs):
        """
        Takes an action in the environment, records the transition, and updates prev_state.
        """
        # Get the current state (self.prev_state is the state before the step)
        state = self.prev_state

        bg = observation.CGM  # Current observation (e.g., blood glucose level)
        meal = kwargs.get('meal')  # unit: g/min

        reward = -(bg - self.target)**2

        # Update the history lists
        self.bg_history.pop(0)
        self.bg_history.append(bg)
        self.meal_history.pop(0)
        self.meal_history.append(meal)

        # Construct state representation for the RL agent
        next_state = torch.tensor(self.bg_history + self.meal_history, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Record the transition (state, action, reward, next_state)
        self.buffer.record((state, action, reward, next_state, done))

        # Update the prev_state to next_state
        self.prev_state = next_state

        # Return some info like done or reward for control
        self.train()

    def train(self):
        """
        Train the actor and critic models using the experience from the replay buffer.
        """
        if self.buffer.buffer_counter < self.batch_size:
            return  # Not enough data to train yet

        # Sample a batch of experiences from the buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample()

        # Compute the target Q-value
        with torch.no_grad():
            target_actions = self.target_actor(next_state_batch)
            # Ensure no future reward is added for terminal states (done_batch=1 for terminal states)
            target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, target_actions) * (1 - done_batch)

        # Update critic
        critic_value = self.critic_model(state_batch, action_batch)
        critic_loss = nn.MSELoss()(critic_value, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor using the sampled policy gradient
        actions = self.actor_model(state_batch)
        actor_loss = -torch.mean(self.critic_model(state_batch, actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of the target networks
        self.update_target_networks()

    def update_target_networks(self):
        """
        Soft updates the target networks using the parameter tau.
        """
        for target_param, param in zip(self.target_actor.parameters(), self.actor_model.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def reset(self, obs, **kwargs):
        """
        Resets the PID controller's internal state.
        """
        bg = obs.CGM  # Current observation (e.g., blood glucose level)
        meal = kwargs.get('meal')  # unit: g/min

        # Initialize history lists
        self.bg_history = [bg] * self.k
        self.meal_history = [meal] * self.k

        # Construct state representation for the RL agent
        self.prev_state = torch.tensor(self.bg_history + self.meal_history, dtype=torch.float32).unsqueeze(0)
