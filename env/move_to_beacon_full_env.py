import math
import numpy as np
import gymnasium as gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, point
from pysc2.lib import viewer as viewer_lib

from env.utils import check_borders, preprocess_channels, calc_step_distance

# pysc2 constants, do not touch
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_UNIT_DENSITY_SCALE = features.SCREEN_FEATURES.unit_density.scale
_PLAYER_FRIENDLY = 1  # marine
_FUNCTIONS = actions.FUNCTIONS

_NUM_DIRECTIONS = 32
_DISCRETE_ACTION_MODE = "discrete"
_CONTINUOUS_ACTION_MODE = "continuous"
_VALID_ACTION_MODES = {_DISCRETE_ACTION_MODE, _CONTINUOUS_ACTION_MODE}


class MoveToBeaconFullEnv(gym.Env):
    def __init__(
        self,
        step_mul=8,
        is_visualize=False,
        enable_web=False,
        web_kwargs=None,
        action_mode=_DISCRETE_ACTION_MODE,
    ):
        if action_mode not in _VALID_ACTION_MODES:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        self.distance = calc_step_distance(step_mul)
        self.feature_screen_size = 32
        self.action_mode = action_mode

        viewer_options = None
        if enable_web:
            viewer_options = self._build_viewer_options(web_kwargs)

        # pysc2 env
        self._env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self.feature_screen_size,
                    minimap=self.feature_screen_size,
                ),
                use_feature_units=True,
                use_camera_position=True,
            ),
            step_mul=step_mul,
            game_steps_per_episode=0,
            visualize=is_visualize,
            viewer_options=viewer_options,
        )

        if self.action_mode == _DISCRETE_ACTION_MODE:
            self.action_space = gym.spaces.Discrete(_NUM_DIRECTIONS)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        low = np.zeros((2, self.feature_screen_size, self.feature_screen_size), dtype=np.int32)
        high = np.stack(
            (
                np.full((self.feature_screen_size, self.feature_screen_size), _PLAYER_RELATIVE_SCALE, dtype=np.int32),
                np.full((self.feature_screen_size, self.feature_screen_size), _UNIT_DENSITY_SCALE, dtype=np.int32),
            )
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)

    @staticmethod
    def _build_viewer_options(web_kwargs=None):
        web_kwargs = dict(web_kwargs or {})
        web_kwargs.setdefault("host", "0.0.0.0")
        web_kwargs.setdefault("port", 8000)
        web_kwargs.setdefault("fps", 15)
        web_kwargs.setdefault("mode", "feature")
        web_kwargs.setdefault("feature_screen", "composite")
        web_kwargs.setdefault("feature_minimap", "composite")
        web_kwargs.setdefault("overlay_units", True)
        web_kwargs.setdefault("overlay_labels", True)
        web_kwargs.setdefault("overlay_camera", True)
        web_kwargs.setdefault("screen_px", point.Point(1280, 960))
        web_kwargs.setdefault("minimap_px", point.Point(320, 320))

        return viewer_lib.ViewerOptions(
            mode="web",
            web=viewer_lib.WebViewerOptions(**web_kwargs),
        )

    def step(self, action):
        self.last_action = self._coerce_action(action)
        if self.marine_pos is None:
            self._set_marine_position()
            if self.marine_pos is None:
                center = self.feature_screen_size / 2
                self.marine_pos = np.array([center, center], dtype=np.float32)

        dx, dy = self._movement_delta(self.last_action)
        new_x, new_y = check_borders(
            self.marine_pos[0] + dx,
            self.marine_pos[1] + dy,
            self.feature_screen_size,
        )
        self.marine_pos = [new_x, new_y]

        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step(
                [_FUNCTIONS.Move_screen("now", [int(round(self.marine_pos[0])), int(round(self.marine_pos[1]))])]
            )[0]
        else:
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        self._set_marine_position()

        self.state = self._get_state()
        self.reward = max(0, self._obs.reward)
        terminated = self._obs.last()
        return self.state, self.reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self._env.reset()[0]
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        self._set_marine_position()

        self.last_action = None
        self.reward = 0
        self.state = self._get_state()
        return self.state, {}

    def _coerce_action(self, action):
        if self.action_mode == _DISCRETE_ACTION_MODE:
            return int(action) % _NUM_DIRECTIONS

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 2:
            raise ValueError(f"Expected continuous action of shape (2,), got {action}")
        action_array = np.clip(action_array, -1.0, 1.0)
        norm = float(np.linalg.norm(action_array))
        if norm > 1.0:
            action_array = action_array / norm
        return action_array

    def _movement_delta(self, action):
        if self.action_mode == _DISCRETE_ACTION_MODE:
            angle = (action % _NUM_DIRECTIONS) * (2 * math.pi / _NUM_DIRECTIONS)
            return math.cos(angle) * self.distance, math.sin(angle) * self.distance
        return float(action[0]) * self.distance, float(action[1]) * self.distance

    def _get_state(self, channels=2):
        state_size = self._obs.observation.feature_screen.shape
        state = np.ndarray(shape=(channels, state_size[1], state_size[2]))
        if channels == 17:
            state = preprocess_channels(self._obs)
        elif channels == 2:
            state[0] = self._obs.observation.feature_screen.player_relative
            state[1] = self._obs.observation.feature_screen.unit_density
        elif channels == 1:
            state[0] = self._obs.observation.feature_screen.unit_density
        return state.astype(np.int32)

    def _set_marine_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if len(marine_x) == 0:
            self.marine_pos = None
            return
        self.marine_pos = np.mean(list(zip(marine_x, marine_y)), axis=0).round()

    def save_replay(self, replay_dir):
        if self._env is not None:
            self._env.save_replay(replay_dir)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def action_shape(self):
        if self.action_mode == _DISCRETE_ACTION_MODE:
            return (int(self.action_space.n),)
        return self.action_space.shape
