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

from itertools import count

import model
import args
import utils

import gym
sys.path.append('/home/mateusz/Documents/AI_Driving_Olympics/duck-driving-golem/gym_duckietown')
import gym_duckietown
from duckietown_rl.wrappers import NormalizeWrapper, ImgWrapper, \
     ActionWrapper, ResizeWrapper
from duckietown_rl.env import launch_env
sys.path.append('/home/mateusz/Documents/AI_Driving_Olympics/duck-driving-golem/src')
from env_with_history import EnvWithHistoryWrapper

# set up matplotlib
is_ipython = 'inline' in mp.get_backend()
if is_ipython:
    from IPython import display

plt.ion() #turn on interactive mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("./parameters"):
    os.makedirs("./parameters")

steps_done = 0
episode_durations = []

#starting env
env = launch_env()

# wrapperino time!
env = ResizeWrapper(env)
#env = EnvWithHistoryWrapper(env,[1,2])
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x6 into 6x160x120
env = ActionWrapper(env)

state_dim = env.observation_space.shape

policy_net = model.DQN(state_dim).to(device)
target_net = model.DQN(state_dim).to(device)

target_net.load_state_dict(policy_net.state_dict())

target_net.eval()

optimizer = op.RMSprop(policy_net.parameters())
memory = model.ReplayMemory(args.MEMEORY_SIZE)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = args.EPS_END + (args.EPS_START - args.EPS_END) * \
        math.exp(-1. * steps_done / args.EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state)
    else:
        return env.action_space.sample()

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


#--------------------------------------------------------

#trainig loopp

def optimize_model():
    if len(memory) < args.BATCH_SIZE:
        return
    transitions = memory.sample(args.BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = model.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.float)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(args.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
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


num_episodes = args.NUM_EPISODES

for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()  #reset env
    obs = torch.from_numpy(obs)
    obs = obs.unsqueeze(0).to(device=device, dtype=torch.float)
    plt.figure()
    plt.imshow(obs.cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
    plt.title('Example extracted screen')
    plt.show()

    # last_screen = get_screen() #previous state
    # current_screen = get_screen() #current state
    # state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = select_action(obs)
        new_obs, reward, done, _ = env.step(action)  #env step here
        new_obs = torch.from_numpy(new_obs)
        new_obs = new_obs.unsqueeze(0).to(device=device, dtype=torch.float)
        reward = torch.tensor([reward], device=device)    

        # Store the transition in memory
        memory.push(obs, action, new_obs, reward)

        # Move to the next state
        obs = new_obs
        obs = obs.unsqueeze(0).to(device)
        obs.to(device=device, dtype=torch.float)
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network
    if i_episode % args.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
print('Saving model to: ./parameters')
target_net.save('args.NUM_NET', './parameters')
print('Parameters saved')

env.render()
env.close()
plt.ioff()
plt.show()