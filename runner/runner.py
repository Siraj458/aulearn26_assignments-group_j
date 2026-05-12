import datetime
import os
from torch.utils.tensorboard import SummaryWriter

from agents.abstractAgent import AbstractAgent

# Used for sliding window averaging calculation
import numpy as np

class Runner:
    def __init__(self, agent, env, is_training=True, save_model_each_episode_num=5, tensorboard_log_dir="./logs", model_save_dir = "./models"):
        self.agent : AbstractAgent = agent
        self.env = env

        self.is_training = is_training
        self.total_score = 0
        self.episodic_score = 0
        self.episode = 1

        self.is_saving_model_during_training = is_training and save_model_each_episode_num > 0 and model_save_dir

        # Path to save: models/agent_name/environment (like move to beacon) NOTE: Had to create the directory manually.
        env_name = type(env).__name__
        if self.is_saving_model_during_training:
            self.models_path = os.path.join(
                os.path.abspath(model_save_dir),
                type(agent).__name__,
                env_name
            )
        # Filename: Datetime + Name + Environment (like move to beacon)
        self.model_file_name = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}' + "_" + str(type(agent).__name__) + "_" + env_name
        self.save_model_each_episode_num = save_model_each_episode_num


        self.tensorboard_log_dir = tensorboard_log_dir
        if tensorboard_log_dir:
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_log_dir),
                type(agent).__name__,
                env_name,
                f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}' + ('_train' if self.is_training else '_eval')
            )
            self.writer = SummaryWriter(self.tb_path)
        
        # Init the circular buffer for the sliding window averaging calculation
        self.window_size = 100
        self.reward_buffer = np.empty(self.window_size)
        self.reward_buffer_i = 0

    def summarize(self):
        # save the model and tensorboard log
        if self.tensorboard_log_dir:
            self.writer.add_scalar('episodic_score', self.episodic_score, self.episode)
            self.writer.add_scalar('total_score', self.total_score, self.episode)
            if hasattr(self.agent, "epsilon"):
                self.writer.add_scalar('epsilon', self.agent.epsilon, self.episode)
            ##### Compute the average reward over a sliding window of 100 episodes #####
            # Write the new episodic score into the circular buffer
            self.reward_buffer[self.reward_buffer_i] = self.episodic_score
            self.reward_buffer_i += 1
            # Jump to the first entry, after passing the last entry:
            if self.reward_buffer_i >= self.window_size:
                self.reward_buffer_i = 0
            # If the amount of values is smaller than the window size, we set the window size to the amount of values we have.
            if self.episode < self.window_size:
                avg_reward_sliding_window = np.sum(self.reward_buffer[0:self.episode]) / self.episode
            else:
                avg_reward_sliding_window = np.sum(self.reward_buffer) / self.window_size
            # Safe the value to tensorboard
            self.writer.add_scalar('avg_reward_sliding_window', avg_reward_sliding_window, self.episode)
            
        if self.is_saving_model_during_training and self.episode % self.save_model_each_episode_num == 0:
            filepath = os.path.join(self.models_path, self.model_file_name)
            self.agent.save_model(filepath)
        # print score
        self.total_score += self.episodic_score
        print(f"Finished episode {self.episode}, episodic score {self.episodic_score}, total score {self.total_score} ")
        self.episode += 1
        self.episodic_score = 0

    def run(self, episodes):
        for _ in range(episodes):
            # get initial state
            state, _ = self.env.reset()
            next_state = None
            done = False
            while not done:
                # get action from the agent
                action = self.agent.get_action(state)
                # get envirionmental response for the performed action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # update the agent with the new experience
                if self.is_training:
                    self.agent.update(state, action, reward, next_state, done)
                # update variables
                state = next_state
                self.episodic_score += reward
            self.summarize()
        self.env.close()
