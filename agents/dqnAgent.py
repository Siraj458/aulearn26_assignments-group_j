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
        # NOTE: For a linear decay, it makes sense to subtract a small amount (e.g. 1 - 0.995 = 0.005)
        self.epsilon_decay = 1 - epsilon_decay
        self.epsilon_min = epsilon_min

        self.target_update_freq = target_update_freq
        self.learn_after_steps = learn_after_steps

        # Initilize our implemented Replay Memory
        self.replay_memory = ReplayMemory(capacity=memory_capacity, batch_size=batch_size)

        # Neural Network for the DQN-Agent depending on the environment
        if discrete_environment:
            self.network = DQN_Discrete(input_size=2, hidden_size=64, output_size=action_shape[0])
            self.target_network = DQN_Discrete(input_size=2, hidden_size=64, output_size=action_shape[0])
        else:
            self.network = DQN_Full(input_size=2, output_size=action_shape[0])
            self.target_network = DQN_Full(input_size=2, output_size=action_shape[0])
        
        # Copy all network parameters to the target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Only the main network is directly trained.
        # The target network just receives the parameters of the main network every 'target_update_freq' iterations.
        # NOTE: The main network is NOT set to train ONLY for the get_action step!
        #       Currently, the network is only trained from replay memory.
        #       BUT maybe there is a good way to also train during get_action()???
        self.network.eval()
        self.target_network.eval()

        # Setup the loss function and the optimizer
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.network.parameters())

        # 8 in discrete_env, 32 in full_env
        self.num_actions = action_shape[0]
        # Count the total amount of update steps
        self.step_count = 0

        # NOTE: Only for testing Network
        #print(self.network.forward(torch.randn(1, 2, 32, 32)))


    

    def get_action(self, state):
        # TODO: Action Selection 

        # Action selection with Epsilon-Greedy (Policy)
        # Explore when below epsilon -> epsilon = exploration chance
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        # Otherwise Exploit instead
        else:
            # Determine max Q-value for this state.
            # The network expects a floating point tensor as an input. (torch.float = torch.float32)
            state_tensor = torch.tensor(state, dtype=torch.float)
            # This calls .forward() implicitly
            # TODO: Think about if we should use network or target_network here!
            #       If we use "target_network", we could keep "network" in training mode the whole time (maybe that can be done anyways?)
            q_values = self.network(state_tensor)

            # NOTE: We have to decide if we still want to do this "smart/random" argmax
            #       or just the default argmax with more exploration in the beginning!!!
            #       ===> since we start with random network weights, it should be HIGHLY unlikely that there are any identical q values in the first place!!!
            ###max_q = torch.max(q_values)
            #### When there are multiple max Q-values, choose a random.
            #### This prevents using always the first index/action, resulting that the Agent is getting stuck at beginning. 
            ###indices = np.where(q_values == max_q)[0]
            ###action = np.random.choice(indices)

            action = torch.argmax(q_values)

        # NOTE: Either 0-7 for discrete or 0-31 for full environment
        return action
    
    def update(self, state, action, reward, next_state, done):
        # TODO: Update logic for training the agent (You could also add transitions)

        # Store transition in replay memory
        # Replay ratio 1/4
        # NOTE: This seemed to make the training worse, but maybe there are situations where this is helpful?
        #if self.step_count % 4 == 0:
        #    self.replay_memory.add(state, action, reward, next_state, done)
        
        #self.replay_memory.add(state, action, reward, next_state, done)
        
        ########## Train the main network ##########
        if self.step_count >= self.learn_after_steps:
            mini_batch = self.replay_memory.sample()

            # TODO: It is inefficient to loop 5 times over the mini_batch! <--- Improve this!
            #       There are many ways to do this. We could also modify the ReplayMemory class if that helps.
            # torch.tensor is faster if we call np.array before, IF we have a list of ndarrays (applies to state & next_state)
            state_tensor = torch.tensor(np.array([i[0] for i in mini_batch]), dtype=torch.float)
            action_tensor = torch.tensor([i[1] for i in mini_batch], dtype=torch.int)
            reward_tensor = torch.tensor([i[2] for i in mini_batch], dtype=torch.float)
            next_state_tensor = torch.tensor(np.array([i[3] for i in mini_batch]), dtype=torch.float)
            done_tensor = torch.tensor([i[4] for i in mini_batch], dtype=torch.float)

            # Set our network to training mode
            self.network.train()
            # Our predictions
            q_values = self.network(state_tensor)
            # Select the q value of the action that was performed at each state transition
            q_values = q_values[torch.arange(q_values.shape[0]), action_tensor]

            # This replaces "q_next = torch.max(self.q_table[next_state_key])" from assignment 1
            # [0] contains the max values and [1] the corresponding indices
            q_values_target = torch.max(self.target_network(next_state_tensor), dim=1)[0]

            # NOTE: There may be more elegant/effictient ways to do implement this logic.
            #       E.g. if we know we are done, we don't need to use the target network for this state transition.
            for i in range(done_tensor.shape[0]):
                if done_tensor[i]:
                    q_values_target[i] = 0

            expected_return = reward_tensor + self.discount_factor * q_values_target

            loss = self.loss_fn(q_values, expected_return)
            self.optimizer.zero_grad()
            loss.backward()
            #running_loss += loss.item()
            self.optimizer.step()

            # Set the network back to eval mode for the next get_action call
            # TODO: Check if this makes sense!
            self.network.eval()

            if self.step_count % self.target_update_freq == 0:
                # Copy all network parameters to the target network
                self.target_network.load_state_dict(self.network.state_dict())
                
        self.step_count += 1

        # Decreasing epsilon (chance of exploring), but only down to the minimum (default 10%)
        #if done:
        #    self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        # During testing it was way better to decrease epsilon down to epsilon min after 1 or 2 episodes...
        # But it was requested that the decay should take half the total episodes to reach epsilon_min!?
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

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
        self.memory.append([state, action, reward, next_state, done])

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
        # Technically the "input layer" is not a linear layer,
        # but just the aplication of the network input to the first hidden layer, without any further processing.
        #self.input = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(input_size, hidden_size) 
        self.hidden2 = nn.Linear(hidden_size, hidden_size)  
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = self.input(x)
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
