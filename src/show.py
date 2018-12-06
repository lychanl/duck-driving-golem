import argparse

import gym
import numpy as np
import pyglet

from gym_duckietown.envs import DuckietownEnv

from src.env_with_history import EnvWithHistoryWrapper



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    args = parser.parse_args()

    if args.env_name is None:
        env = DuckietownEnv(
            map_name = args.map_name,
            draw_curve = args.draw_curve,
            draw_bbox = args.draw_bbox,
            domain_rand = args.domain_rand,
            frame_skip = args.frame_skip,
            distortion = args.distortion,
        )
    else:
        env = gym.make(args.env_name)

    env = EnvWithHistoryWrapper(env, 3, 10)

    env.reset()
    env.render()

    def update(dt):
        env.render('human')
        env.step(np.array([0.1, 0.0], dtype=np.float32))

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

    try:
        # Enter main event loop
        pyglet.app.run()
    except:
        pass

    env.close()


if __name__ == '__main__':
    main()
