import numpy as np

from agents.abstractAgent import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    Random agent
    """

    def __init__(self, observation_space, action_shape):
        super().__init__(observation_space, action_shape)

    def get_action(self, state):
        if self.action_shape == (2,):
            return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return int(np.random.randint(self.action_shape[0]))

    def update(self, state, action, reward, next_state, done):
        pass

    def save_model(self, _):
        raise NotImplementedError("Error: Random Agent is stateless!")

    def load_model(self, _):
        raise NotImplementedError("Error: Random Agent is stateless!")
