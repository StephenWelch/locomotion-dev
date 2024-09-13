# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import gymnasium as gym
import os
import random
import time
from dataclasses import dataclass

from omni.isaac.lab.app import AppLauncher
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def add_args(parser: argparse.ArgumentParser):
    # Algorithm specific arguments
    parser.add_argument("--exp_name",                type=str,    default=os.path.basename(__file__)[: -len(".py")],
                        help="the name of this experiment")
    parser.add_argument("--seed",                    type=int,    default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch_deterministic",     action="store_true",
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda",                    action="store_true",
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track",                   action="store_true",
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name",      type=str,    default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb_entity",            type=str,    default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video",           action="store_true",
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--env_id",                  type=str,    default="Ant",
                        help="the id of the environment")
    parser.add_argument("--total_timesteps",         type=int,    default=30000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate",           type=float,  default=0.0026,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs",                type=int,    default=4096,
                        help="the number of parallel game environments")
    parser.add_argument("--num_steps",               type=int,    default=16,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr",               action="store_true",
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma",                   type=float,  default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae_lambda",              type=float,  default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches",         type=int,    default=2,
                        help="the number of mini-batches")
    parser.add_argument("--update_epochs",           type=int,    default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv",                action="store_true",
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef",               type=float,  default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss",              action="store_true",
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef",                type=float,  default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef",                 type=float,  default=2,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm",           type=float,  default=1,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl",               type=float,  default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--reward_scaler",           type=float,  default=1,
                        help="the scale factor applied to the reward during training")
    parser.add_argument("--record_video_step_frequency", type=int, default=1464,
                        help="the frequency at which to record the videos")
    
    # Checkpointing
    parser.add_argument("--save_interval",           type=int,    default=100,
                        help="the interval to save a checkpoint")

    # RPO
    parser.add_argument("--rpo_alpha",              type=float,  default=0.0,
                        help="the alpha parameter for RPO") 
    parser.add_argument("--noise_exp",              type=float,  default=0.0,)

    # to be filled in runtime
    parser.add_argument("--batch_size",              type=int,    default=0,
                        help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch_size",          type=int,    default=0,
                        help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num_iterations",          type=int,    default=0,
                        help="the number of iterations (computed in runtime)")

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, rpo_alpha, num_obs, num_actions, num_critic_obs=None):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        if num_critic_obs is None:
            num_critic_obs = num_obs
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_critic_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, envs.num_actions), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.num_actions))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            # z = self.rpo_alpha*powerlaw_psd_gaussian(1.0, action_mean.shape, device=device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Continuous Action IsaacLab")
    add_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app 

    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    import locomotion_dev.tasks  # noqa: F401
    from locomotion_dev.colored_noise import powerlaw_psd_gaussian

    env_cfg = parse_env_cfg(
        args.env_id, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    envs = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array" if args.capture_video else None)

    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    envs = RslRlVecEnvWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    
    obs, extras = envs.get_observations()
    num_obs = obs.shape[1]
    if "critic" in extras["observations"]:
        num_critic_obs = extras["observations"]["critic"].shape[1]
    else:
        num_critic_obs = num_obs

    envs.single_action_space = gym.spaces.Box(-float("inf"), float("inf"), shape=(envs.num_actions,), dtype=np.float32)
    envs.single_observation_space = gym.spaces.Box(-float("inf"), float("inf"), shape=(num_obs,), dtype=np.float32)

    # envs.single_action_space = envs.action_space
    # envs.single_observation_space = envs.observation_space
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(args.rpo_alpha, num_obs, envs.num_actions, num_critic_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    for iteration in range(1, args.num_iterations + 1):
        iter_start_time = time.time()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                        if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                            writer.add_scalar(
                                "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                            )
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break


        # Checkpoint
        if iteration % args.save_interval:
            torch.save(agent.state_dict(), f"runs/{run_name}/agent_{global_step}.pth")

        print(f"RSL SPS: {int(args.num_steps * args.num_envs / (time.time() - iter_start_time))}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        

    # envs.close()
    writer.close()