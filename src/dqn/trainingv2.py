import sys, os
import random
import math

import torch
import torch.optim as op
import torch.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import gym

import gym_duckietown
import model
from duckietown_rl import args
from duckietown_rl.wrappers import ActionWrapper, ResizeWrapper, DeltasToActionsWrapper
from duckietown_rl.utils import choose_action
from duckietown_rl.env import launch_env
sys.path.append('/home/mateusz/Documents/AI_Driving_Olympics/duck-driving-golem/src')
from env_with_history import EnvWithHistoryWrapper

# set up matplotlib
is_ipython = 'inline' in mp.get_backend()
if is_ipython:
    from IPython import display


#----------------------------
def epsThreshold(steps_done):
    return args.EPS_END + (args.EPS_START - args.EPS_END) * \
        math.exp(-1. * steps_done / args.EPS_DECAY)
#----------------------------

random.seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#parameters folder
if not os.path.exists("./parameters"):
    os.makedirs("./parameters")

#start env
env = launch_env()

#wrappers
env = ResizeWrapper(env)
env = EnvWithHistoryWrapper(env, 2, 5)
print(env.action_space.shape)
env = DeltasToActionsWrapper(env, delta_vel=args.VELOCITY_DELTA, delta_omega=args.ANGLE_DELTA)
print(env.action_space.shape)
env = ActionWrapper(env)
print(env.action_space.shape)

#initialaziing nets
policy_net = model.DQN().to(device)
frozen_net = model.DQN().to(device)

optimizer = op.Adam(policy_net.parameters())

memory = model.ReplayMemory(args.MEMORY_SIZE)

totensor = T.ToTensor()


done = False
episode_durations=[]

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


#training loop
for episode in range(args.NUM_EPISODES):
    steps_done=0
    obs = totensor(env.reset()).unsqueeze(0).to(device)
    while not done:
        steps_done += 1
        print('\n \n Step: ', steps_done)
        action=choose_action(steps_done,policy_net, obs)
        print('Action: ', action, 'Type: ', type(action))
        print('Performing action...')
        next_obs, reward, done, _ = env.step(action)

        print('Done, reward: ',reward, type(reward))
        next_obs = totensor(next_obs).unsqueeze(0).to(device)
        reward = torch.tensor([reward], device=device)
        action = torch.from_numpy(action)
        
        print('Types of obs, action, next_obs, reward: ',type(obs),type(action),type(next_obs),type(reward),'\n')
        memory.push(obs, action, next_obs, reward)
        obs = next_obs

        if len(memory)>args.BATCH_SIZE:
            batch = memory.sample(args.BATCH_SIZE)
            print('\n Memory sampled')

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)#.to(device)
            action_batch = torch.cat(batch.action)#.to(device)
            reward_batch = torch.cat(batch.reward)#.to(device)
            print('Batch collected')

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = policy_net(state_batch)
            
            print('Q(s_t,a)')
            print(state_action_values)
            
            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(args.BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = torch.t(frozen_net(non_final_next_states))
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * args.GAMMA) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
        
        if steps_done % args.TARGET_UPDATE == 0:
            frozen_net.load_state_dict(policy_net.state_dict())

    episode_durations.append(steps_done + 1)
    plot_durations()

print('Complete')
print('Saving model to: ./parameters')
frozen_net.save('args.NUM_NET', './parameters')
print('Parameters saved')