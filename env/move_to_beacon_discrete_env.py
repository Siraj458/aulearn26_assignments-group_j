import numpy as np
import gymnasium as gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, point
from pysc2.lib import viewer as viewer_lib

from env.utils import discretize_distance, calc_target_position, calc_step_distance

# pysc2 constants, do not touch
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1  # marine
_PLAYER_NEUTRAL = 3  # beacon
_FUNCTIONS = actions.FUNCTIONS

_DISCRETE_ACTION_MODE = "discrete"
_CONTINUOUS_ACTION_MODE = "continuous"
_VALID_ACTION_MODES = {_DISCRETE_ACTION_MODE, _CONTINUOUS_ACTION_MODE}

# fix the actions for 8 directions
ACTION_DIRECTION = {
    0: "up_left",
    1: "up",
    2: "up_right",
    3: "right",
    4: "down_right",
    5: "down",
    6: "down_left",
    7: "left",
}

_ACTION_VECTORS = {
    0: np.asarray([-1.0, -1.0], dtype=np.float32),
    1: np.asarray([0.0, -1.0], dtype=np.float32),
    2: np.asarray([1.0, -1.0], dtype=np.float32),
    3: np.asarray([1.0, 0.0], dtype=np.float32),
    4: np.asarray([1.0, 1.0], dtype=np.float32),
    5: np.asarray([0.0, 1.0], dtype=np.float32),
    6: np.asarray([-1.0, 1.0], dtype=np.float32),
    7: np.asarray([-1.0, 0.0], dtype=np.float32),
}


class MoveToBeaconDiscreteEnv(gym.Env):
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

        self.distance_discrete_range = 10
        self.distance = calc_step_distance(step_mul)
        self.feature_screen_size = 32
        self.action_mode = action_mode
        self.marine_pos = None
        self.beacon_pos = None

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
            self.action_space = gym.spaces.Discrete(len(ACTION_DIRECTION))
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-self.distance_discrete_range,
            high=self.distance_discrete_range,
            shape=(2,),
            dtype=np.int32,
        )

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

        new_x, new_y = self._target_for_action(self.last_action)
        target_position = [int(round(new_x)), int(round(new_y))]

        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step([_FUNCTIONS.Move_screen("now", target_position)])[0]
        else:
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        self._set_marine_position()
        self._set_beacon_position()

        self.state = self._get_state()
        self.reward = self._obs.reward
        terminated = self._obs.last()
        return self.state, self.reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self._env.reset()[0]
        # select the army once after reset
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        self._set_marine_position()
        self._set_beacon_position()

        self.last_action = None
        self.reward = 0
        self.state = self._get_state()
        return self.state, {}

    def _coerce_action(self, action):
        if self.action_mode == _DISCRETE_ACTION_MODE:
            return int(action) % len(ACTION_DIRECTION)

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 2:
            raise ValueError(f"Expected continuous action of shape (2,), got {action}")
        action_array = np.clip(action_array, -1.0, 1.0)
        norm = float(np.linalg.norm(action_array))
        if norm > 1.0:
            action_array = action_array / norm
        return action_array

    def _target_for_action(self, action):
        if self.marine_pos is None:
            center = self.feature_screen_size / 2
            self.marine_pos = np.asarray([center, center], dtype=np.float32)

        if self.action_mode == _DISCRETE_ACTION_MODE:
            if self._should_snap_to_beacon(action):
                return self.beacon_pos[0], self.beacon_pos[1]

            return calc_target_position(
                self.marine_pos[0],
                self.marine_pos[1],
                ACTION_DIRECTION[action],
                self.distance,
                self.feature_screen_size,
            )

        dx = float(action[0]) * self.distance
        dy = float(action[1]) * self.distance
        return calc_target_position(
            self.marine_pos[0],
            self.marine_pos[1],
            "right" if dx >= 0 else "left",
            abs(dx),
            self.feature_screen_size,
        )[0], calc_target_position(
            self.marine_pos[0],
            self.marine_pos[1],
            "down" if dy >= 0 else "up",
            abs(dy),
            self.feature_screen_size,
        )[1]

    def _should_snap_to_beacon(self, action):
        if self.beacon_pos is None or self.marine_pos is None:
            return False

        discrete_offset = self._get_state()
        if np.max(np.abs(discrete_offset)) > 1:
            return False

        action_vector = _ACTION_VECTORS[int(action)]
        desired_direction = np.sign(discrete_offset.astype(np.float32))
        if float(np.dot(action_vector, desired_direction)) <= 0.0:
            return False

        for axis in range(2):
            action_sign = int(np.sign(action_vector[axis]))
            desired_sign = int(np.sign(desired_direction[axis]))
            if action_sign != 0 and desired_sign != 0 and action_sign != desired_sign:
                return False
        return True

    def _set_marine_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if len(marine_x) == 0:
            return
        self.marine_pos = np.mean(list(zip(marine_x, marine_y)), axis=0).round()

    def _set_beacon_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        beacon_y, beacon_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        if len(beacon_x) == 0:
            return
        self.beacon_pos = np.mean(list(zip(beacon_x, beacon_y)), axis=0).round()

    def _get_state(self):
        if self.marine_pos is None or self.beacon_pos is None:
            return np.zeros(2, dtype=np.int32)

        x_distance = self.beacon_pos[0] - self.marine_pos[0]
        y_distance = self.beacon_pos[1] - self.marine_pos[1]
        return np.array(
            [
                discretize_distance(
                    x_distance,
                    self.feature_screen_size,
                    self.distance_discrete_range,
                ),
                discretize_distance(
                    y_distance,
                    self.feature_screen_size,
                    self.distance_discrete_range,
                ),
            ],
            dtype=np.int32,
        )

    def roll_to_next_state(self, action):
        if self.action_mode != _DISCRETE_ACTION_MODE:
            raise RuntimeError("roll_to_next_state is only available in discrete action mode")
        new_x, new_y = calc_target_position(
            self.marine_pos[0],
            self.marine_pos[1],
            ACTION_DIRECTION[action],
            self.distance,
            self.feature_screen_size,
        )
        x_distance = self.beacon_pos[0] - new_x
        y_distance = self.beacon_pos[1] - new_y
        return np.array([x_distance, y_distance], dtype=np.int32)

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
