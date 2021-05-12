import os
import random
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic with Sampling Strategies')
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 5 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter determines the relative importance of \
                          the entropy term against the reward\
                          0.2 for HalfCeetah, Hopper, Ant, Walker2d\
                          0.05 for Humanoid')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 512)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden dimension of neural networks (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--use_writer', type=bool, default=False)
parser.add_argument('--mode', default='SAC', help = 'SAC, EREe, EREo, ERE2, or HAR')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))

env = gym.make(args.env_name)
print(args.env_name)
print(env.observation_space.shape[0], env.action_space.shape[0])

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
env.seed(args.seed)
env.action_space.seed(args.seed)

agent = SAC(env.observation_space.shape[0], env.action_space, args)

#TesnorboardX
if args.use_writer:
    writer = SummaryWriter(logdir='runs/{}_{}_{}'.format(args.mode, args.env_name, args.seed))

# Memory
memory = ReplayMemory(args.replay_size)

# Evaluation
def evaluate(agent, env, episodes=10, save_traj=False):
    if save_traj:
        traj=[]

    returns = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action = agent.select_action(state, eval=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            
            if save_traj:
                traj.append( (state, action, reward, next_state, mask) )

            state = next_state
        returns[i] = episode_reward

    if not save_traj:
        return returns.mean(), returns.std()
    else:
        return traj

# Training Loop
total_numsteps = 0
updates = 0
if not args.use_writer:
    fout = open("result/{}_{}_{}.csv".format(args.mode, args.env_name, args.seed), 'w')
        
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and args.mode!='EREo':
            # Update parameters of all the networks
            log_likeli, dist = agent.update_parameters(memory, args, None, episode_steps) 
            updates += 1
            if total_numsteps%20 == 0 and args.use_writer:
                writer.add_scalar('log_likeli', log_likeli.item(), total_numsteps)
                writer.add_scalar('dist', dist.item(), total_numsteps)

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    memory.update_trajlen(episode_steps)

    if args.mode == 'EREo':
        for k in range(episode_steps):
            log_likeli, dist = agent.update_parameters(memory, args, episode_steps, k) 

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 5 == 0 and args.eval == True:
        episodes = 10
        avg_reward, std = evaluate(agent, env, episodes=episodes, save_traj=False)

        if args.use_writer:
            writer.add_scalar('avg_reward', avg_reward, total_numsteps)
            writer.flush()
        else:
            fout.write('{},{},{}\n'.format(total_numsteps, avg_reward, std))
            fout.flush()

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
