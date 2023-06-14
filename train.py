import argparse
import os
import time

import torch

from pathlib import Path
from gym.wrappers import FrameStack, TransformObservation

from agent import Agent
from dino import Dino
from model import BasicCNN, BasicLSTM
from utils import MetricLogger, SkipFrame, load_latest_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--capacity', type=int, default=1000)
parser.add_argument('--episodes', type=int, default=10000)
parser.add_argument('--epsilon-decay', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--soft_update', default=False, action='store_true')
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--update_interval', type=int, default=1000)
parser.add_argument('--logdir', type=str, default='logs')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = Dino()
env = TransformObservation(env, lambda x: x / 255.)
env = SkipFrame(env, 2)
env = FrameStack(env, 4)

logger = MetricLogger(Path('./logs'))

state, info = env.reset()
n_observations = state.shape[1]
n_actions = env.action_space.n
policy_net = BasicLSTM(n_observations, n_actions).to(device)
target_net = BasicLSTM(n_observations, n_actions).to(device)

agent = Agent(n_actions, policy_net, target_net, args, device)
# agent.load_checkpoint('./logs/model_1000.pth')
load_latest_checkpoint(args.logdir, agent)

for e in range(args.episodes):

    state, info = env.reset()

    while True:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.remember(state, next_state, action, reward, terminated)

        q, loss = agent.learn()

        logger.step(reward, loss, q)

        state = next_state

        if terminated or truncated:
            time.sleep(1)
            break

    logger.episode(info)
    if e != 0 and e % 1000 == 0:
        agent.save_checkpoint(f'./logs/model_{e}.pth')
    if e % 10 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
