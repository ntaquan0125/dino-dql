import os
import re
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from glob import glob


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        Return only every `skip`-th frame
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Repeat action, and sum reward
        """
        total_reward = 0.0
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return observation, reward, terminated, truncated, info


def load_latest_checkpoint(logdir, agent):
    checkpoint_files = glob(os.path.join(logdir, '*.pth'))
    if len(checkpoint_files) > 0:
        checkpoint_files = sorted(checkpoint_files, key=lambda s: int(re.search(r'\d+', s).group()))
        print('Found latest checkpoint at ' + checkpoint_files[-1])
        agent.load_checkpoint(checkpoint_files[-1])
    else:
        print('No checkpoint')



class MetricLogger:
    def __init__(self, save_dir):
        self.ep_rewards_plot = save_dir / 'reward_plot.jpg'
        self.ep_losses_plot = save_dir / 'loss_plot.jpg'
        self.ep_qs_plot = save_dir / 'q_plot.jpg'

        # History metrics
        self.ep_rewards = []
        self.ep_scores = []
        self.ep_losses = []
        self.ep_qs = []

        # Plot history
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_losses = []
        self.moving_avg_ep_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def step(self, reward, loss, q):
        self.ep_reward += reward
        if loss:
            self.step_losses.append(loss)
            self.step_qs.append(q)

    def episode(self, info):
        self.ep_rewards.append(self.ep_reward)
        self.ep_scores.append(info['score'])
        self.ep_losses.append(np.mean(self.step_losses))
        self.ep_qs.append(np.mean(self.step_qs))

        self.init_episode()

    def init_episode(self):
        self.ep_reward = 0.0
        self.step_losses = []
        self.step_qs = []

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_score = np.round(np.mean(self.ep_scores[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_losses.append(mean_ep_loss)
        self.moving_avg_ep_qs.append(mean_ep_q)

        self.ep_rewards = []
        self.ep_scores = []
        self.ep_losses = []
        self.ep_qs = []

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f'Episode: {episode} - '
            f'Step: {step} - '
            f'Epsilon: {epsilon} - '
            f'Reward: {mean_ep_reward} - '
            f'Score: {mean_ep_score} - '
            f'Loss: {mean_ep_loss} - '
            f'Q Value: {mean_ep_q} - '
            f'Time: {time_since_last_record}'
        )

        for metric in ['ep_rewards', 'ep_losses', 'ep_qs']:
            plt.plot(getattr(self, f'moving_avg_{metric}'))
            plt.savefig(getattr(self, f'{metric}_plot'))
            plt.clf()