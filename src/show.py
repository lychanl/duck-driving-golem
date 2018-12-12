import os

import numpy as np
import pyglet
import png

from src.env import create_env_from_args


def _save_image(img, path, name_prefix, step):
    name = os.path.join(path, "{}_{}.png".format(name_prefix, step))
    png.from_array(img, mode='RGB').save(name)


def save_images(env, actor, nsteps, path, name_prefix):
    obs = env.reset()

    for step in range(nsteps):
        img = env.render('rgb_array')
        action = actor(obs)
        obs, _, _, _ = env.step(action)
        _save_image(img, path, name_prefix, step)


def show(env, actor, nsteps):
    step = 0

    obs = env.reset()
    done = False

    env.render()

    def update(dt):
        env.render('human')

        nonlocal obs, done

        if not done:
            action = actor(obs)
            obs, _, done, _ = env.step(action)

        if nsteps:
            nonlocal step
            step += 1

            if step >= nsteps:
                pyglet.clock.unschedule(update)
                pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

    pyglet.app.run()


def main():
    env = create_env_from_args()

    show(env, lambda _: np.array([0.1, -0.2], dtype=np.float32), None)

    env.close()


if __name__ == '__main__':
    main()
