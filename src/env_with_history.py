import numpy as np
from gym.spaces import Box
from gym import Env

from gym import Wrapper


class EnvWithHistoryWrapper(Env):
    def __init__(self, wrapped, input_frames, frequency):
        super(EnvWithHistoryWrapper, self).__init__()
        self.wrapped = wrapped
        self.history = [None for _ in range(input_frames * frequency)]
        self.history_iter = 0

        self.input_frames = input_frames
        self.frequency = frequency

        self.action_space = wrapped.action_space

        shape = (
                    wrapped.observation_space.shape[0],
                    wrapped.observation_space.shape[1],
                    wrapped.observation_space.shape[2] * input_frames)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8
        )

    @property
    def unwrapped(self):
        return self.wrapped.unwrapped

    def step(self, action):
        obs, reward, done, info = self.wrapped.step(action)

        frames = []
        for i in range(self.input_frames - 1):
            frames.append(self.history[(self.history_iter + i * self.frequency) % len(self.history)])

        frames.append(obs)

        obs_with_history = np.concatenate(frames, axis=2)

        self.history[self.history_iter] = obs
        self.history_iter = (self.history_iter + 1) % len(self.history)

        return obs_with_history, reward, done, info

    def reset(self):
        obs = self.wrapped.reset()

        for i in range(len(self.history)):
            self.history[i] = obs

        return np.concatenate([obs for _ in range(self.input_frames)], axis=2)

    def render(self, mode='human'):
        return self.wrapped.render(mode)

    def close(self):
        return self.wrapped.close()

    def seed(self, seed=None):
        return self.wrapped.seed(seed)


class VecEnvWithHistoryFactory:
    """
    A factory to use as environment-creating function in vectorized environments
    """
    def __init__(self, wrapped_fun, input_frames, frequency):
        self.wrapped_fun = wrapped_fun
        self.input_frames = input_frames
        self.frequency = frequency

    def __call__(self, *args, **kwargs):
        wrapped = self.wrapped_fun(*args, **kwargs)
        return EnvWithHistoryWrapper(wrapped, self.input_frames, self.frequency)
