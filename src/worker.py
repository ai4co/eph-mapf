import os
import random
import subprocess
import threading
import time

from copy import deepcopy
from typing import Tuple

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from hydra.utils import instantiate
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.buffer import EpisodeData, LocalBuffer, SumTree
from src.config import config
from src.environment import Environment

Network = instantiate({"_target_": config.model_target, "_partial_": True})


if config.use_wandb:
    os.environ["RAY_DEDUP_LOGS"] = "0"  # disable RAY_DEDUP_LOGS=0


# Define the command execution function and mark it as a Ray remote function
@ray.remote
def execute_command(command):
    try:
        subprocess.run(command)
    except subprocess.CalledProcessError:
        pass


@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(
        self,
        buffer_capacity=config.buffer_capacity,
        init_env_settings=tuple(config.init_env_settings),
        alpha=config.prioritized_replay_alpha,
        beta=config.prioritized_replay_beta,
        chunk_capacity=config.chunk_capacity,
    ):

        self.capacity = buffer_capacity
        self.chunk_capacity = chunk_capacity
        self.num_chunks = buffer_capacity // chunk_capacity
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings: []}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = [None] * self.num_chunks
        self.last_act_buf = [None] * self.num_chunks
        self.act_buf = np.zeros((buffer_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((buffer_capacity), dtype=np.float16)
        self.hid_buf = [None] * self.num_chunks
        self.size_buf = np.zeros(self.num_chunks, dtype=np.uint8)
        self.relative_pos_buf = [None] * self.num_chunks
        self.comm_mask_buf = [None] * self.num_chunks
        self.gamma_buf = np.zeros((self.capacity), dtype=np.float16)
        self.num_agents_buf = np.zeros((self.num_chunks), dtype=np.uint8)

        if config.use_wandb:
            wandb.init(
                project=config.project,
                name=config.name,
                id=config.run_id,
                config=dict(config),
            )
            self.wandb = wandb

    def __len__(self):
        return np.sum(self.size_buf)

    def run(self):
        self.background_thread = threading.Thread(target=self._prepare_data, daemon=True)
        self.background_thread.start()

    def _prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)

            else:
                time.sleep(0.1)

    def sample_batch(self, *args, **kwargs):  # NOTE: NEW, for avoiding bugs
        return self._sample_batch(*args, **kwargs)

    def get_batched_data(self):
        """
        get one batch of data, called by learner.
        """

        if len(self.batched_data) == 0:
            print("no prepared data")
            data = self._sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: EpisodeData):
        """
        Add one episode data into replay buffer, called by actor if actor finished one episode.

        data: actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5,
                hid_buf 6, comm_mask_buf 8, gamma 9, td_errors 10, sizes 11, done 12
        """
        if data.actor_id >= 9:  # eps-greedy < 0.01
            stat_key = (data.num_agents, data.map_len)
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data.done)
                if len(self.stat_dict[stat_key]) == config.cl_history_size + 1:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:

            for i, size in enumerate(data.sizes):
                idxes = np.arange(
                    self.ptr * self.chunk_capacity, (self.ptr + 1) * self.chunk_capacity
                )
                start_idx = self.ptr * self.chunk_capacity
                # update buffer size
                self.counter += size

                self.priority_tree.batch_update(
                    idxes,
                    data.td_errors[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                    ]
                    ** self.alpha,
                )

                self.obs_buf[self.ptr] = np.copy(
                    data.obs[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.last_act_buf[self.ptr] = np.copy(
                    data.last_act[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.act_buf[start_idx : start_idx + size] = data.actions[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.rew_buf[start_idx : start_idx + size] = data.rewards[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.hid_buf[self.ptr] = np.copy(
                    data.hiddens[
                        i * self.chunk_capacity : i * self.chunk_capacity
                        + size
                        + config.forward_steps
                    ]
                )
                self.size_buf[self.ptr] = size
                self.relative_pos_buf[self.ptr] = np.copy(
                    data.relative_pos[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.comm_mask_buf[self.ptr] = np.copy(
                    data.comm_mask[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.gamma_buf[start_idx : start_idx + size] = data.gammas[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.num_agents_buf[self.ptr] = data.num_agents

                self.ptr = (self.ptr + 1) % self.num_chunks

            del data

    def _sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_last_act, b_steps, b_relative_pos, b_comm_mask = [], [], [], [], []
        b_hidden = []
        idxes, priorities = [], []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.chunk_capacity
            local_idxes = idxes % self.chunk_capacity
            max_num_agents = np.max(self.num_agents_buf[global_idxes])

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):

                assert (
                    local_idx < self.size_buf[global_idx]
                ), "index is {} but size is {}, p {}".format(
                    local_idx, self.size_buf[global_idx], self.priority_tree[idx]
                )

                steps = min(
                    config.forward_steps, self.size_buf[global_idx].item() - local_idx
                )

                relative_pos = self.relative_pos_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                comm_mask = self.comm_mask_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                obs = self.obs_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                last_act = self.last_act_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                hidden = self.hid_buf[global_idx][local_idx]

                if steps < config.forward_steps:
                    pad_len = config.forward_steps - steps
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, pad_len), (0, 0), (0, 0)))
                    relative_pos = np.pad(
                        relative_pos, ((0, pad_len), (0, 0), (0, 0), (0, 0))
                    )
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0)))

                if self.num_agents_buf[global_idx] < max_num_agents:
                    pad_len = max_num_agents - self.num_agents_buf[global_idx].item()
                    obs = np.pad(obs, ((0, 0), (0, pad_len), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, 0), (0, pad_len), (0, 0)))
                    relative_pos = np.pad(
                        relative_pos, ((0, 0), (0, pad_len), (0, pad_len), (0, 0))
                    )
                    comm_mask = np.pad(comm_mask, ((0, 0), (0, pad_len), (0, pad_len)))
                    hidden = np.pad(hidden, ((0, pad_len), (0, 0)))

                b_obs.append(obs)
                b_last_act.append(last_act)
                b_steps.append(steps)
                b_relative_pos.append(relative_pos)
                b_comm_mask.append(comm_mask)
                b_hidden.append(hidden)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities / min_p, -self.beta)

            b_action = self.act_buf[idxes]
            b_reward = self.rew_buf[idxes]
            b_gamma = self.gamma_buf[idxes]

            data = (
                torch.from_numpy(np.stack(b_obs)).transpose(1, 0).contiguous(),
                torch.from_numpy(np.stack(b_last_act)).transpose(1, 0).contiguous(),
                torch.from_numpy(b_action).unsqueeze(1),
                torch.from_numpy(b_reward).unsqueeze(1),
                torch.from_numpy(b_gamma).unsqueeze(1),
                torch.ByteTensor(b_steps),
                torch.from_numpy(np.concatenate(b_hidden, axis=0)),
                torch.from_numpy(np.stack(b_relative_pos)),
                torch.from_numpy(np.stack(b_comm_mask)),
                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.ptr,
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr * self.chunk_capacity) | (
                    idxes >= self.ptr * self.chunk_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.chunk_capacity) & (
                    idxes >= self.ptr * self.chunk_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(
                np.copy(idxes), np.copy(priorities) ** self.alpha
            )

    def stats(self, interval: int):
        """
        Print log
        """
        print("buffer update speed: {}/s".format(self.counter / interval))
        print("buffer size: {}".format(np.sum(self.size_buf)))

        print("  ", end="")
        for i in range(config.init_env_settings[1], config.max_map_lenght + 1, 5):
            print("   {:2d}   ".format(i), end="")
        print()

        for num_agents in range(config.init_env_settings[0], config.max_num_agents + 1):
            # for num_agents in range(config.init_env_settings[0], config.max_num_agents+1, 4):
            print("{:2d}".format(num_agents), end="")
            for map_len in range(
                config.init_env_settings[1], config.max_map_lenght + 1, 5
            ):
                if (num_agents, map_len) in self.stat_dict:
                    print(
                        "{:4d}/{:<3d}".format(
                            sum(self.stat_dict[(num_agents, map_len)]),
                            len(self.stat_dict[(num_agents, map_len)]),
                        ),
                        end="",
                    )
                else:
                    print("   N/A  ", end="")
            print()

        # Wandb loggin
        if config.use_wandb:
            self.wandb.log({"buffer_size": np.sum(self.size_buf)})
            self.wandb.log({"buffer_update_speed": self.counter / interval})
            for num_agents in range(
                config.init_env_settings[0], config.max_num_agents + 1
            ):
                # for num_agents in range(config.init_env_settings[0], config.max_num_agents+1, 4):
                for map_len in range(
                    config.init_env_settings[1], config.max_map_lenght + 1, 5
                ):
                    fmt = "Agents#{:02d}/Map#{:02d}"
                    if (num_agents, map_len) in self.stat_dict:
                        self.wandb.log(
                            {
                                "train/"
                                + fmt.format(num_agents, map_len): sum(
                                    self.stat_dict[(num_agents, map_len)]
                                )
                                / (len(self.stat_dict[(num_agents, map_len)]) + 1e-10)
                            }
                        )
                    else:
                        self.wandb.log({"train/" + fmt.format(num_agents, map_len): 0})

        for key, val in self.stat_dict.copy().items():
            # print('{}: {}/{}'.format(key, sum(val), len(val)))
            if (
                len(val) == config.cl_history_size
                and sum(val) >= config.cl_history_size * config.pass_rate
            ):
                # add number of agents
                add_agent_key = (key[0] + 1, key[1])
                # add_agent_key = (key[0]+4, key[1])
                if (
                    add_agent_key[0] <= config.max_num_agents
                    and add_agent_key not in self.stat_dict
                ):
                    self.stat_dict[add_agent_key] = []

                if key[1] < config.max_map_lenght:
                    add_map_key = (key[0], key[1] + 5)
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []

        self.env_settings_set = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False

    def get_env_settings(self):
        return self.env_settings_set


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.lr_scheduler_milestones,
            gamma=config.lr_scheduler_gamma,
        )
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0

        self.data_list = []

        self.store_weights()

        if config.use_wandb:
            wandb.init(
                project=config.project,
                name=config.name,
                id=config.run_id,
                config=dict(config),
            )

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self._train, daemon=True)
        self.learning_thread.start()

    def _train(self):
        scaler = GradScaler()
        b_seq_len = torch.LongTensor(config.batch_size)
        b_seq_len[:] = config.burn_in_steps + 1

        checkpoint_path = None

        for i in range(1, config.training_steps + 1):

            data_id = ray.get(self.buffer.get_batched_data.remote())
            data = ray.get(data_id)

            (
                b_obs,
                b_last_act,
                b_action,
                b_reward,
                b_gamma,
                b_steps,
                b_hidden,
                b_relative_pos,
                b_comm_mask,
                idxes,
                weights,
                old_ptr,
            ) = data
            b_obs, b_last_act, b_action, b_reward = (
                b_obs.to(self.device),
                b_last_act.to(self.device),
                b_action.to(self.device),
                b_reward.to(self.device),
            )
            b_gamma, weights = b_gamma.to(self.device), weights.to(self.device)
            b_hidden = b_hidden.to(self.device)
            b_relative_pos, b_comm_mask = b_relative_pos.to(self.device), b_comm_mask.to(
                self.device
            )

            b_action = b_action.long()

            b_obs, b_last_act = b_obs.half(), b_last_act.half()

            b_next_seq_len = b_seq_len + b_steps

            with torch.no_grad():
                b_q_ = self.tar_model(
                    b_obs,
                    b_last_act,
                    b_next_seq_len,
                    b_hidden,
                    b_relative_pos,
                    b_comm_mask,
                ).max(1, keepdim=True)[0]

            target_q = b_reward + b_gamma * b_q_

            b_q = self.model(
                b_obs[: -config.forward_steps],
                b_last_act[: -config.forward_steps],
                b_seq_len,
                b_hidden,
                b_relative_pos[:, : -config.forward_steps],
                b_comm_mask[:, : -config.forward_steps],
            ).gather(1, b_action)

            td_error = target_q - b_q

            priorities = (
                td_error.detach().clone().squeeze().abs().clamp(1e-6).cpu().numpy()
            )

            loss = F.mse_loss(b_q, target_q)
            self.loss += loss.item()

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_dqn)
            scaler.step(self.optimizer)
            scaler.update()

            self.scheduler.step()

            # store new weights in shared memory
            if i % 2 == 0:
                self.store_weights()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            self.counter += 1

            # update target net, save model
            if i % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())

            if i % config.save_interval == 0:
                # create save path if not exist
                if not os.path.exists(config.save_path):
                    os.makedirs(config.save_path)
                checkpoint_path = os.path.join(config.save_path, f"{self.counter}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

            if i % config.val_interval == 0:
                assert (
                    checkpoint_path is not None
                ), "A checkpoint path must be saved before validation"
                self.validate(checkpoint_path, self.counter, config.run_id, config.name)

        self.done = True

    def validate(self, checkpoint_path, step, run_id, name):
        """Run subprocess in background and detach directly"""
        print("Validation subprocess started")
        command = [
            "python",
            "validate.py",
            "--checkpoint_path",
            checkpoint_path,
            "--step",
            str(step),
            "--run_id",
            run_id,
            "--name",
            name,
        ]
        print(" ".join(command))

        # Note: took me forever to get this working
        # subprocess does not work apparently, so need to do stuff with Ray
        future = execute_command.remote(command)

    def stats(self, interval: int):
        """
        print log
        """
        print("number of updates: {}".format(self.counter))
        print("update speed: {}/s".format((self.counter - self.last_counter) / interval))
        if self.counter != self.last_counter:
            print("loss: {:.4f}".format(self.loss / (self.counter - self.last_counter)))

        if config.use_wandb:
            wandb.log({"update speed": (self.counter - self.last_counter) / interval})
            if self.counter != self.last_counter:
                wandb.log({"loss": self.loss / (self.counter - self.last_counter)})
            else:
                wandb.log({"loss": 0})

            wandb.log({"number of updates": self.counter})

        self.last_counter = self.counter
        self.loss = 0
        return self.done


@ray.remote(num_cpus=1)
class Actor:
    def __init__(
        self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer
    ):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment(curriculum=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0

    def run(self):
        done = False
        obs, last_act, pos, local_buffer = self._reset()

        while True:

            # sample action
            actions, q_val, hidden, relative_pos, comm_mask = self.model.step(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(last_act.astype(np.float32)),
                torch.from_numpy(pos.astype(int)),
            )

            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, config.action_dim)

            # take action in env
            (next_obs, last_act, next_pos), rewards, done, _ = self.env.step(actions)
            # return data and update observation
            local_buffer.add(
                q_val[0],
                actions[0],
                last_act,
                rewards[0],
                next_obs,
                hidden,
                relative_pos,
                comm_mask,
            )

            if done == False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, _, relative_pos, comm_mask = self.model.step(
                        torch.from_numpy(next_obs.astype(np.float32)),
                        torch.from_numpy(last_act.astype(np.float32)),
                        torch.from_numpy(next_pos.astype(int)),
                    )
                    data = local_buffer.finish(q_val[0], relative_pos, comm_mask)

                self.global_buffer.add.remote(data)
                done = False
                obs, last_act, pos, local_buffer = self._reset()

            self.counter += 1
            if self.counter == config.actor_update_steps:
                self._update_weights()
                self.counter = 0

    def _update_weights(self):
        """load weights from learner"""
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def _reset(self):
        self.model.reset()
        obs, last_act, pos = self.env.reset()
        local_buffer = LocalBuffer(
            self.id, self.env.num_agents, self.env.map_size[0], obs
        )
        return obs, last_act, pos, local_buffer
