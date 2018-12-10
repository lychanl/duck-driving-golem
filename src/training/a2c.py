import os
import sys

import tensorflow as tf
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.a2c.a2c as a2c
from baselines.common import tf_util
from baselines.common.policies import build_policy

from src.env import get_env_creator_from_args
from src.show import show, save_images
from src.models.a2c_cnn import a2c_discrete_cnn


def train(env, save_path, nsteps=20, timesteps=1e3):
    model = a2c.learn(a2c_discrete_cnn, env, nsteps=nsteps, total_timesteps=int(timesteps),
                      load_path=save_path if os.path.isfile(save_path) else None)
    model.save(save_path)


def test(env, load_path, img_path, display_steps=500):
    with tf.Session() as sess:
        policy = build_policy(env, a2c_discrete_cnn)
        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            model = policy(1, 1, sess)

        tf_util.load_variables(load_path, sess=sess)

        def display_actor(obs):
            actions = model.step([obs])[0]
            return actions[0]

        if img_path is None:
            show(env, display_actor, display_steps)
        else:
            save_images(env, display_actor, display_steps, img_path, 'img_')


def main():
    env_creator = get_env_creator_from_args()

    model_file = sys.argv[1]

    if sys.argv[2] == 'train':
        nenvs = 5
        steps = int(sys.argv[3])

        env = SubprocVecEnv([env_creator for _ in range(nenvs)])
        train(env=env, save_path=model_file, timesteps=steps)
    elif sys.argv[2] == 'test':
        testenv = env_creator()

        test(testenv, model_file, None if len(sys.argv) < 4 else sys.argv[3])


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
