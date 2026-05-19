import numpy as np
import os
import torch
import random
from collections import deque
from torch import nn

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
                learn_after_steps=6000):
        super().__init__(observation_space, action_shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # TODO: Stuff is missing here


    
    def get_action(self, state):
       # TODO: Action Selection 

        # NOTE: Just for testing. Action space is now continous from -1 to 1 of shape (2,) -> the direction
        return ((1,-1))
    
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
        # TODO: Init me!
        pass

    def add(self, state, action, reward, next_state, done):
        # TODO: Add transitions
        pass

    def sample(self):
        # TODO: Implement sampling from the replay memory
        pass

    def __len__(self):
        return len(self.memory)