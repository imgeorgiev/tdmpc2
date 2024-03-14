from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from common.logger import dict_mean

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._fps_time = None

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        tot_time = time() - self._start_time
        d = dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=tot_time,
        )
        if self._fps_time:
            tot_time = time() - self._fps_time
            d.update({"fps": self._step // tot_time})
        return d

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
            action = torch.full_like(self.env.rand_act()[0], float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).to("cuda")
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
        train_metrics = {}
        done = torch.tensor([True] * self.cfg.env.num_envs)
        pretrained = False
        ep_rewards = torch.zeros(self.cfg.env.num_envs, device="cuda")
        obs = self.env.reset()
        self._tds = [None] * self.cfg.env.num_envs

        while self._step <= self.cfg.steps:
            if done.any():

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

                # if "dflex" not in self.cfg.task:
                #     obs = self.env.reset()

                for idx in done.nonzero(as_tuple=False):
                    self._tds[idx] = [self.to_td(obs[idx].squeeze())]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                # NOTE: planning doesn't work with vectorized envs
                action = self.agent.act(obs, t0=done.any())
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)

            with torch.no_grad():
                ep_rewards += reward
                # NOTE: this can be probably be done in a more efficient manner
                for i, (o, a, r) in enumerate(zip(obs, action, reward)):
                    if done[i]:
                        o = info["obs_before_reset"][i]
                    self._tds[i].append(self.to_td(o, a, r))

            # Once we are done seding, pretrain the agent
            if self._step >= self.cfg.seed_steps and not pretrained:
                print("Pretraining agent on seed data...")
                metrics = []
                for k in range(self.cfg.seed_train_steps):
                    print(f"Update {k}/{self.cfg.seed_train_steps}", end="\r")
                    _train_metrics = self.agent.update(self.buffer)
                    metrics.append(_train_metrics)

                train_metrics.update(dict_mean(metrics))
                pretrained = True
                self._fps_time = time()  # start tracking fps

            # Once we are done pre-training, train the agent at every step
            if pretrained:
                _train_metrics = self.agent.update(self.buffer)
                metrics.append(_train_metrics)

            self._step += 1 * self.cfg.env.num_envs

        print("Evaluation")
        eval_metrics = self.eval()
        eval_metrics.update(self.common_metrics())
        self.logger.log(eval_metrics, "eval")

        self.logger.finish(self.agent)
