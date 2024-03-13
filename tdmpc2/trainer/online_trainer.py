from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
import torch
from tqdm import tqdm
from common.logger import dict_mean

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        tot_time = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=tot_time,
            fps=self._step // tot_time,
        )

    @torch.no_grad()
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        ep = 0
        obs, done, ep_reward, t = self.env.reset(), False, 0, 0
        warm_start = True
        while ep < self.cfg.eval_episodes:

            # NOTE: broken due to vectorization
            # if self.cfg.save_video:
            #     self.logger.video.init(self.env, enabled=(ep == 0))
            action = self.agent.act(obs, t0=warm_start, eval_mode=True)
            obs, reward, done, info = self.env.step(action)
            ep_reward += reward
            t += 1
            if warm_start:
                warm_start = False

            if torch.any(done):
                ep += torch.sum(done).item()
                ep_rewards.extend(ep_reward[done].tolist())
                ep_successes.extend(info["success"][done].tolist())
            # if self.cfg.save_video:
            #     self.logger.video.record(self.env)
            # ep_rewards.append(ep_reward.cpu().numpy())
            # ep_successes.append(info["success"])
            # if self.cfg.save_video:
            # self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0)
        if action is None:
            action = torch.full_like(self.env.rand_act()[0].unsqueeze(0), float("nan"))
        if reward is None:
            reward = torch.tensor([float("nan")]).to("cuda")
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = (
            {},
            torch.tensor([True] * self.cfg.env.num_envs),
            True,
        )
        pretrained = False
        train_steps = self.cfg.env.num_envs * self.cfg.train_freq
        last_train_step = 0
        ep_rewards = torch.zeros(self.cfg.env.num_envs, device="cuda")
        self._tds = [None] * self.cfg.env.num_envs
        obs = self.env.reset()

        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done.any():
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=ep_rewards[done].mean().item(),
                        episode_success=info["success"].float().mean().item(),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    ep_rewards[done] = 0
                    for idx in done.nonzero(as_tuple=False):
                        if len(self._tds[idx]) > self.cfg.horizon + 1:
                            self._ep_idx = self.buffer.add(torch.cat(self._tds[idx]))
                        else:
                            print(f"WARN: Trajectory too short {len(self._tds[idx])}")

                if "dflex" not in self.cfg.task:
                    obs = self.env.reset()

                for idx in done.nonzero(as_tuple=False):
                    with torch.no_grad():
                        self._tds[idx] = [self.to_td(obs[idx])]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=done.any())
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)

            with torch.no_grad():
                ep_rewards += reward
                for i, (o, a, r) in enumerate(zip(obs, action, reward)):
                    if done[i]:
                        o = info["obs_before_reset"][i]
                    o = o.unsqueeze(0)
                    a = a.unsqueeze(0)
                    r = r.unsqueeze(0)
                    self._tds[i].append(self.to_td(o, a, r))

            # Update agent
            if self._step >= self.cfg.seed_steps and self._ep_idx > 0:
                num_updates = 0

                if self._step - last_train_step > train_steps:
                    last_train_step = self._step
                    num_updates = self.cfg.num_updates

                if not pretrained:
                    num_updates = self.cfg.seed_steps // self.cfg.env.num_envs
                    pretrained = True
                    print("Pretraining agent on seed data...")

                metrics = []
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                    metrics.append(_train_metrics)

                if len(metrics) > 0:
                    train_metrics.update(dict_mean(metrics))

            self._step += 1 * self.cfg.env.num_envs

        self.logger.finish(self.agent)
