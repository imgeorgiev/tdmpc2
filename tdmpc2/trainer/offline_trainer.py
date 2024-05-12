import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from common import TASK_SET
from trainer.base import Trainer


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
                while not done:
                    action = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, task=task_idx
                    )
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
            results.update(
                {
                    f"episode_reward+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_rewards
                    ),
                    f"episode_success+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_successes
                    ),
                }
            )
        return results

    def eval1(self):
        """Evaluate a TD-MPC2 agent on a single env."""
        results = dict()
        ep_rewards, ep_successes = [], []
        for _ in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), torch.tensor([False]), 0, 0
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True, task=None)
                obs, reward, done, info = self.env.step(action.cpu())
                ep_reward += reward.item()
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
        results.update(
            {
                f"episode_reward": np.nanmean(ep_rewards),
                f"episode_success": np.nanmean(ep_successes),
            }
        )
        return results

    def train(self):
        """Train a TD-MPC2 agent."""
        # assert self.cfg.multitask and self.cfg.task in {
        #     "mt30",
        #     "mt80",
        # }, "Offline training only supports multitask training with mt30 or mt80 task sets."

        # Load data
        # assert self.cfg.task in self.cfg.data_dir, (
        #     f"Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, "
        #     f"please double-check your config."
        # )
        fp = Path(os.path.join(self.cfg.data_dir, "*.pt"))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f"No data found at {fp}"
        print(f"Found {len(fps)} files in {fp}")

        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task == "mt80" else 501
        # _cfg.buffer_size = 550_450_000 if self.cfg.task == "mt80" else 345_690_000
        _cfg.buffer_size = len(fps) * 32_768 * _cfg.episode_length
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        # fps = fps[:3]
        for fp in tqdm(fps, desc="Loading data"):
            td = torch.load(fp).to("cuda")
            # convert td to tensor dict
            # filter out the tasks that are not 17 and save as torch tensor
            assert td.shape[1] == _cfg.episode_length, (
                f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                f"please double-check your config."
            )
            # find all td indices where all tasks are 17 @todo, change to use variable task_id
            task_idxs = torch.where(torch.all(td["task"] == 17, dim=1))[0]
            print("number of episodes with task 17 (hopper-stand):", len(task_idxs))
            num_episodes = 0
            for i in task_idxs:
                if (num_episodes == 1000):
                    break
                print(f"Episode {num_episodes}/{len(task_idxs)}", end="\r")
                self.buffer.add(td[i])
                num_episodes += 1
        # assert (
        #     self.buffer.num_eps == self.buffer.capacity
        # ), f"Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes."

        print(f"Training agent for {self.cfg.steps} iterations...")
        metrics = {}
        for i in range(self.cfg.steps):
            print(f"Step {i}/{self.cfg.steps}", end="\r")

            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 50 == 0:
                metrics = {
                    "iteration": i,
                    "total_time": time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval1())
                    # self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        print("saving agent")
                        self.logger.save_agent(self.agent, identifier=f"{i}")
                self.logger.log(metrics, "pretrain")

                if i % self.cfg.save_policy_freq == 0:
                    print("saving policy")
                    self.logger.save_policy(self.agent, identifier=f"{i}")

        self.logger.finish(self.agent)
