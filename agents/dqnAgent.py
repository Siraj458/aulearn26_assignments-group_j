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
                discrete_environment=True): # NOTE new parameter for setting this agent up in the depending environment
        super().__init__(observation_space, action_shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # NOTE: For a linear decay, it makes sense to subtract a small amount (e.g. 1 - 0.995 = 0.005)
        self.epsilon_decay = 1 - epsilon_decay
        self.epsilon_min = epsilon_min

        self.discrete_environment = discrete_environment
        # How often the target-network is updated with the weights of the Q-network
        self.target_update_freq = target_update_freq
        # If the target-network should be softly updated each step
        self.tau = 0.005
        # NOTE: Important that training only starts when there are enough samples for a batch! (total steps > batch_size)
        self.learn_after_steps = learn_after_steps 

        # Initialize our implemented Replay Memory
        self.replay_memory = ReplayMemory(capacity=memory_capacity, batch_size=batch_size)
        # Logic for a Replay-Ratio (Not training every step OR multiple trainings per step)
        self.replay_ratio = 1
        self.accumulated_ratio = 0

        # Neural Network for the DQN-Agent depending on the environment
        if discrete_environment:
            # NOTE: For the smaller network (discrete env) cpu could work even faster
            self.device = torch.device('cpu')
            self.q_network = DQN_Discrete(input_size=2, hidden_size=64, output_size=action_shape[0]).to(self.device)
            self.target_network = DQN_Discrete(input_size=2, hidden_size=64, output_size=action_shape[0]).to(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.q_network = DQN_Full(input_size=2, output_size=action_shape[0]).to(self.device)
            self.target_network = DQN_Full(input_size=2, output_size=action_shape[0]).to(self.device)
        
        # Copy all network parameters to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Only the main network is directly trained.
        # The target network just receives the parameters of the main network every 'target_update_freq' iterations.
        self.target_network.eval()

        # Setup the loss function and the optimizer
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.learning_rate)

        # 8 in discrete_env, 32 in full_env
        self.num_actions = action_shape[0]
        # Count the total amount of update steps
        self.step_count = 0



    # TODO: LOOK AT https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

    def get_action(self, state):
        """Action selection with Epsilon-Greedy (Policy)"""
        # NOTE: Either 0-7 for discrete or 0-31 for full environment

        # Explore when below epsilon -> epsilon = exploration chance
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
       
        # Otherwise Exploit instead
        else:
            # Determine max Q-value for this state.
            # The network expects a floating point tensor as an input. (torch.float = torch.float32)
            # The full environment network expects one extra dimension for the batch size
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            #print("State Tensor Shape:", state_tensor.shape)

            # NOTE: From simply using the forward pass of the network, we won't train the weights.
            #       HOWEVER whe need to disable some layer functions like Dropout or Batch-Norm with .eval().
            #       AND PyTorch keeps track of the calculations, to make later gradient computation more easily.
            #       By disabling that with no_grad(), we save us some memory.  
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor) # This calls .forward() implicitly
            action = torch.argmax(q_values)
    
        return action
    


    def update(self, state, action, reward, next_state, done):
        """Update logic for training the agent"""

        # Store every transition in replay memory 
        self.replay_memory.add(state, action, reward, next_state, done)

        # NOTE: Logic for Replay-Ratio. When:
        #       Ratio < 1  -->  train the Q-network not every update/step (can be usefull if training is expensive)
        #       Ratio = 1  -->  train every step (standard)
        #       Ratio > 1  -->  train the Q-network multiple times per step (when creating data for a step is expensive, e.g. intensive simulation)
        # A too large ratio tends to let the network overfit with the experiences saved in the replay memory.
        self.accumulated_ratio += self.replay_ratio 


        # ========== Train the main Q-network ==========
        # Train as long we have enough accumulate ratios:
        # (ratio = 0.25 --> every 4th step)
        # (ratio = 4.0  --> 4 times per step)
        while self.accumulated_ratio >= 1.0:

            # Only train, if we have enough samples for one batch.
            if self.step_count >= self.learn_after_steps:
                self.train_model()

            self.accumulated_ratio -= 1.0

        # ========== Update the target-network ==========
        # We NEVER train the target-network, as it has to deliver stable target-values for our Q-network to train with.
        # However, it has to be updated after some time in order to deliver target-values from future steps that actually has meanings.
        # Otherwise our Q-network would never learn actually effective strategies.

        # Hard-Update of target-network (copy all network parameters every x steps)
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
                
        # NOTE Alternative: Soft-Update every step (very slowly adjusting the target-network weights towards the Q-network weights)
        # tau = self.tau # 0.005
        # # For every pair of parameter (weights or bias), let the Q-network have a small impact on the parameters of the target-network.
        # for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
        #     # .parameters() returns pointers, and copy_() is from PyTorch to replaces those pure parameters with this new value.
        #     target_network_param.data.copy_(tau * q_network_param.data + (1.0 - tau) * target_network_param.data)
        # ===============================================


        self.step_count += 1
        # Decreasing epsilon (chance of exploring), but only down to the minimum (default 10%)
        # NOTE: During testing it was way better to decrease epsilon down to epsilon min after 1 or 2 episodes
        # But it was requested that the decay should take half the total episodes to reach epsilon_min
        if done:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)


    # td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten()) ????

    def train_model(self):
        """Sample a batch from the replay memory and train the Q-network."""
        self.q_network.train()
        # From sample() we get a list
        mini_batch = self.replay_memory.sample()
        # The * unzips the outer list from sample(), while the zip() zips the content element wise into new lists.
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # torch.tensor is faster if we call np.array before, IF we have a list of ndarrays (applies to state & next_state)
        state_tensor = torch.tensor(np.array(states), dtype=torch.float, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float, device=self.device)


        # Our predictions
        # Gather only the q-value of the action that was performed at each state transition and corresponds to the reward
        # squeeze() removes given dimensions of size 1. unsqueeze() would ad a dimension of size 1
        # For gather `state_tensor` and `action_tensor` must have the same shape. 
        # Later our `q_values` and `target_values` must have same shape --> important to correctly compute the loss
        q_values = self.q_network(state_tensor).gather(1, index=action_tensor.unsqueeze(1)).squeeze()
        
        # This replaces "q_next = torch.max(self.q_table[next_state_key])" from assignment 1
        # NOTE: We don't want to train the target_network
        with torch.no_grad():
            # [0] contains the max values and [1] the corresponding indices
            q_values_target = torch.max(self.target_network(next_state_tensor), dim=1)[0]

        # print("Q-Values Shape:", q_values.shape)
        # print("Targets Shape:", q_values_target.shape)
        # NOTE: There may be more elegant/effictient ways to do implement this logic.
        #       E.g. if we know we are done, we don't need to use the target network for this state transition.
        for i in range(done_tensor.shape[0]):
            if done_tensor[i]:
                q_values_target[i] = 0

        expected_return = reward_tensor + self.discount_factor * q_values_target

        # Loss between our current Q-values and the target Q-values.
        loss = self.loss_fn(q_values, expected_return)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def save_model(self, path, filename="dqn.pt"):
        """Save the learnable parameters of your model and the hyperparameters"""
        # Also save the obersavtion space and action shape -> Thus we can recreate the whole agent upon loading.
        # NOTE: We don't save the replay memory, as it is generally not recommended and costs space.
        #       Simply generate new data when continuing training.
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'target_update_freq': self.target_update_freq,
            'learn_after_steps': self.learn_after_steps,
            'discrete_environment': self.discrete_environment, # determines type of network
            'observation_space': self.observation_space,
            'action_shape': self.action_shape,
        }
        torch.save(checkpoint, path + f"/{filename}")


    @classmethod
    def load_model(cls, path, filename="dqn.pt", eval_mode=False, reset_timesteps=False, load_memory=True):
        """
        Load the learnable parameters of your model and the hyperparameters
        Afterwards instantiated a DQN-agent with those parameters
        """
        checkpoint = torch.load(path + f"/{filename}")
        if eval_mode:
            agent = cls(checkpoint['observation_space'], 
                        checkpoint['action_shape'], 
                        batch_size=checkpoint['batch_size'],
                        learning_rate=checkpoint['learning_rate'],
                        discount_factor=checkpoint['discount_factor'],
                        epsilon=0.1, 
                        epsilon_min=0.1, # just a small epsilon to prevent getting stuck
                        target_update_freq=checkpoint['target_update_freq'],
                        learn_after_steps=checkpoint['learn_after_steps'],
                        discrete_environment=checkpoint['discrete_environment']) 
        else:
            agent = cls(checkpoint['observation_space'], 
                        checkpoint['action_shape'],
                        batch_size=checkpoint['batch_size'],
                        learning_rate=checkpoint['learning_rate'],
                        discount_factor=checkpoint['discount_factor'],
                        epsilon=checkpoint['epsilon'],  
                        epsilon_decay=checkpoint['epsilon_decay'],  
                        epsilon_min=checkpoint['epsilon_min'], 
                        target_update_freq=checkpoint['target_update_freq'],
                        learn_after_steps=checkpoint['learn_after_steps'],
                        discrete_environment=checkpoint['discrete_environment']) 

        # Loading the networks and optimizer
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        agent.target_network.load_state_dict(checkpoint['target_network_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return agent


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
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)  
        self.out = nn.Linear(256, output_size)  

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), kernel_size=8)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        #print("Shape after flatten (for input size): ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x) # Output without activation
        return x
