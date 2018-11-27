import numpy as np
from gym.spaces import Box
from gym import Env


class EnvWithHistoryWrapper(Env):
    def __init__(self, wrapped, history_range):
        super(EnvWithHistoryWrapper, self).__init__()
        self.wrapped = wrapped
        self.history = [None for _ in history_range]
        self.history_iter = 0

        self.action_space = wrapped.action_space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(wrapped.camera_height, wrapped.camera_width, 6),
            dtype=np.uint8
        )

    def step(self, action):
        obs, reward, done, info = self.wrapped.step(action)

        historic_obs = self.history[self.history_iter]
        obs_with_history = np.concatenate((obs, historic_obs), axis=2)

        self.history[self.history_iter] = obs
        self.history_iter = (self.history_iter + 1) % len(self.history)

        return obs_with_history

    def reset(self):
        obs = self.wrapped.reset()

        for i in range(self.history_iter):
            self.history[i] = obs

        return np.concatenate((obs, obs), axis=2)

    def render(self, mode='human'):
        return self.wrapped.render(mode)

    def close(self):
        return self.wrapped.close()

    def seed(self, seed=None):
        return self.wrapped.seed(seed)
