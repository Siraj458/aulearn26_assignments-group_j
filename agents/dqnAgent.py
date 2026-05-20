import numpy as np
import os
import torch
import random
from collections import deque
from torch import nn
import torch.nn.functional as F

from agents.abstractAgent import AbstractAgent

class DQNAgent(AbstractAgent):
    """
    A DQN model that uses neural networks to approximate the value function for Q-Learning.
    """
    def __init__(self, observation_space, action_shape, 
                batch_size=32, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                net_arch=[64, 64], 
                target_update_freq=10, 
                memory_capacity=10000,
                learn_after_steps=6000,
                discrete_environment=True): # New parameter for setting this agent up in the depending environment
        super().__init__(observation_space, action_shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initilize our implemented Replay Memory
        self.replay_memory = ReplayMemory(capacity=memory_capacity, batch_size=batch_size)

        # Neural Network for the DQN-Agent depending on the environment
        if discrete_environment:
            self.network = DQN_Discrete(input_size=2, hidden_size=64, output_size=action_shape[0])
        else:
            self.network = DQN_Full(input_size=2, output_size=action_shape[0])
        
        # NOTE: Only for testing Network
        #print(self.network.forward(torch.randn(1, 2, 32, 32)))

    

    def get_action(self, state):
       # TODO: Action Selection 


        # NOTE: Either 0-7 for discrete or 0-31 for full environment
        return 30
    
    def update(self, state, action, reward, next_state, done):
        # TODO: Update logic for training the agent (You could also add transitions)
        pass

    def save_model(self, path, filename="dqn.pt"):
        # TODO: Save the learnable parameters of your model and the hyperparameters
        # torch.save should be helpful for this :)
        pass

    @classmethod
    def load_model(cls, path, filename="dqn.pt", eval_mode=False, reset_timesteps=False, load_memory=True):
        # TODO: Load the learnable parameters of your model and the hyperparameters
        # torch.load should be helpful for this :)
        # Afterwards you have to instantiated a DQN agent with those parameters
        pass



class ReplayMemory:
    """
    Replay memory for off-policy agents: a simple buffer and a PER version
    """
    def __init__(self, capacity, batch_size, is_prioritized=False, alpha=0.6, beta=0.4, beta_annealing=0.9999):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        #self.is_prioritized = is_prioritized # TODO: See how this should be used
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        # TODO: Implement sampling from the replay memory
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
    


class DQN_Discrete(nn.Module):
    """
    Neural Network for the DQN-Agent in the discrete environment with discrete action-space (8).
    """
    def __init__(self, input_size:int=2, hidden_size:int=64, output_size:int=8):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size) 
        self.hidden2 = nn.Linear(hidden_size, hidden_size)  
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x) # Output without activation
        return x

class DQN_Full(nn.Module):
    """
    Neural Network for the DQN-Agent in the full environment with discrete action-space (32).
    """
    def __init__(self, input_size:int=2, output_size:int=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)  
        self.out = nn.Linear(256, output_size)  

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), kernel_size=2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        #print("Shape after flatten (for input size): ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x) # Output without activation
        return x
