import numpy as np

from agents.abstractAgent import AbstractAgent


_DIRECTION_FROM_SIGNS = {
    (-1, -1): 0,  # up_left
    (0, -1): 1,   # up
    (1, -1): 2,   # up_right
    (1, 0): 3,    # right
    (1, 1): 4,    # down_right
    (0, 1): 5,    # down
    (-1, 1): 6,   # down_left
    (-1, 0): 7,   # left
}


class BasicAgent(AbstractAgent):
    """
    Basic dummy agent that selects the action based on the straightforward angle calculation, no RL
    """

    def __init__(self, observation_space, action_shape):
        super().__init__(observation_space, action_shape)
        self._last_discrete_action = None

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        if self.action_shape == (2,):
            norm = float(np.linalg.norm(state))
            if norm == 0.0:
                return np.zeros(2, dtype=np.float32)
            return (state / norm).astype(np.float32)

        state = np.rint(state).astype(np.int32)
        x_sign = int(np.sign(state[0]))
        y_sign = int(np.sign(state[1]))

        if x_sign == 0 and y_sign == 0:
            if self._last_discrete_action is not None:
                return self._last_discrete_action
            return 1

        action = _DIRECTION_FROM_SIGNS[(x_sign, y_sign)]
        self._last_discrete_action = action
        return action

    def update(self, state, action, reward, next_state, done):
        pass

    def save_model(self, _):
        raise NotImplementedError("Error: Basic Agent is stateless!")

    @classmethod
    def load_model(cls, _):
        raise NotImplementedError("Error: Basic Agent is stateless!")
