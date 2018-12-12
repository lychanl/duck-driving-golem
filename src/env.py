import argparse

import gym
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from gym_duckietown.wrappers import  DiscreteWrapper

from src.env_with_history import VecEnvWithHistoryFactory


def get_env_creator_from_args(discrete=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--map-name', default='loop_empty')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    args, _ = parser.parse_known_args()

    def creator():
        if args.env_name is None:
            env = DuckietownEnv(
                map_name=args.map_name,
                draw_curve=args.draw_curve,
                draw_bbox=args.draw_bbox,
                domain_rand=args.domain_rand,
                frame_skip=args.frame_skip,
                distortion=args.distortion,
            )
        else:
            env = gym.make(args.env_name)

        return DiscreteWrapper(env) if discrete else env

    return VecEnvWithHistoryFactory(creator, 3, 10)


def create_env_from_args():
    return get_env_creator_from_args()()
