import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


class Agent:
    def __init__(self, action_dim, policy_net, target_net, config, device='cuda'):
        self.action_dim = action_dim
        self.policy_net = policy_net
        self.target_net = target_net
        self.config = config
        self.memory = deque(maxlen=self.config.capacity)
        self.batch_size = self.config.batch_size
        self.device = device
        self.curr_step = 0
        self.EPSILON_START = 1.0
        self.EPSILON_END = 0.1
        self.exploration_rate = self.EPSILON_START

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)


    def select_action(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step
        """
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
            action_values = self.policy_net(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) \
            * np.exp(-1. * self.curr_step * self.config.epsilon_decay)
        self.curr_step += 1

        return action_idx

    def remember(self, state, next_state, action, reward, done):
        """
        Store the experience to replay buffer
        """
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        states, next_states, actions, rewards, dones = map(torch.stack, zip(*batch))
        return states, next_states, actions, rewards, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None

        states, next_states, actions, rewards, dones = self.recall()

        current_Q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_state_Q = self.policy_net(next_states)
            best_actions = torch.argmax(next_state_Q, axis=1).unsqueeze(-1)
            next_Q = self.target_net(next_states).gather(1, best_actions)

        expected_Q = (rewards + (1 - dones.float()) * self.config.gamma * next_Q).float()
        loss = self.criterion(current_Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        if self.config.soft_update:
            self.soft_update()
        else:
            self.hard_update()

        return current_Q.mean().item(), loss.item()

    def hard_update(self):
        if self.curr_step % self.config.update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        # θ′ ← τ*θ + (1 - τ)*θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.tau \
                + target_net_state_dict[key] * (1 - self.config.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_checkpoint(self, save_path):
        print('Saving checkpoint at ' + save_path)
        torch.save(
            dict(
                model=self.policy_net.state_dict(),
                optimizer=self.optimizer.state_dict(),
                exploration_rate=self.exploration_rate,
                step=self.curr_step
            ),
            save_path
        )

    def load_checkpoint(self, load_path):
        print('Loading checkpoint at ' + load_path)
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['model'])
        self.target_net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['step']