import numpy as np
import pandas as pd
import pickle
import os
from tensordict import TensorDict
import torch

from agents.abstractAgent import AbstractAgent

class QLearningAgent(AbstractAgent):
    """
    Q Learning RL agent that produces actions based on the highest Q value, computes and saves sa-pairs in a Q table
    """
    def __init__(self, observation_space, action_shape,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.1):

        super().__init__(observation_space, action_shape)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Create the Q-table, BUT initilize it dynamically for each state when needed.
        self.q_table = TensorDict({})
        # 8 in our case, used later for Q-table
        self.num_actions = action_shape[0]




    def get_action(self, state):
        # When state has no entry in Q-table yet, initilize a Tensor with number of actions as length
        # resulting in a state-action-pair.
        state_key = str(state)
        if not state_key in self.q_table:
            # TensorDict uses strings as keys. States are discrete (rounded integer) here.
            self.q_table[state_key] = torch.zeros(self.num_actions) + 0.5 # + np.random.random()


        # Action selection with Epsilon-Greedy (Policy)
        # Explore when below epsilon -> epsilon = exploration chance
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        # Otherwise Exploit instead
        else:
            # Determine max Q-value for this state.
            q_values = self.q_table[state_key]
            max_q = torch.max(q_values)
            # When there are multiple max Q-values, choose a random.
            # This prevents using always the first index/action, resulting that the Agent is getting stuck at beginning. 
            indices = np.where(q_values == max_q)[0]
            action = np.random.choice(indices)
            # Alternatively: Just pick the first max value index (worse learning perfomance!)
            #action = indices[0]

        # Decreasing epsilon (chance of exploring), but only down to the minimum (default 10%)
        # NOTE: We chose to do this here and we later noticed that the epsilon value would decay down to ~0.3 after just one episode!
        #       We also tested much slower decay rates which reached the min value after about 250 episodes.
        #       Alternatively, we could apply the decay once after each episode, but our method
        #       should be fine, as long as the decay value is adjusted accordingly!
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action



    def update(self, state, action, reward, next_state, done):
        # TensorDict uses strings as keys.
        state_key = str(state)
        q = self.q_table[state_key][action]

        # When done, there is no next state and q-value
        if done:
            q_next = 0
        else:
            # As we looking for the next state, we need to check if it has an entry in the table
            next_state_key = str(next_state)
            if not next_state_key in self.q_table:
                self.q_table[next_state_key] = torch.zeros(self.num_actions) + 0.5 # + np.random.random()

            # NOTE: Q-Learning
            # Max Q-value of the next state (off-policy) -> expecting to use the best action next
            q_next = torch.max(self.q_table[next_state_key])


        # Sum of rewards of this and next state (also named G or target), depending on Q-value of next step. 
        # Using discount factor to reduce the future q-values with each step, making the reward more valuable with fewer steps.
        # Otherwise reward would be same even if agent takes infinite steps - mathematically converges to infinity
        expected_return = reward + self.discount_factor * q_next

        # Updating Q-value for this state-action-pair, with parts of the current and next Q-value.
        # Alpha determines how slow the new information impacts the current one -> preventing jumping Q-values while exploring
        # (expected_return - q) is also called Temporal Difference 
        self.q_table[state_key][action] = q + self.learning_rate * (expected_return - q)



    def save_model(self, filepath):
        # We save not only the Q-table but also the obersavtion space and action shape -> Thus we can recreate the whole agent upon loading.
        fileContents = (self.q_table, self.observation_space, self.action_shape)
        # Using pickle is better to handle complex types, custom classes and nested structures
        # 'wb' is used for binary writing
        print(f"Filepath = {filepath}")
        with open(filepath + '.pkl', 'wb') as file:
            pickle.dump(fileContents, file)


    @classmethod
    def load_model(cls, filepath, eval_mode):  # NOTE: Had to include eval_mode
        # Inverse of save_model - load the pickled file
        with open(filepath, 'rb') as file:
            q_table, observation_space, action_shape = pickle.load(file)
        
        # Recreate the agent
        # In evaluation mode, exploring without learning is meaningless, so we can set epsilon to zero -> Choosing action by exploiting.
        if eval_mode:
            agent = cls(observation_space, action_shape, epsilon=0, epsilon_min=0)
        else:
            agent = cls(observation_space, action_shape)
            
        agent.q_table = q_table
        return agent

