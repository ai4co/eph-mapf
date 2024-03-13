import torch.multiprocessing as mp
import pickle
import torch
from typing import Tuple
# from model import Network
import numpy as np
import os
import wandb
import time
import random

from src.environment import Environment
from src.config import config

# NOTE: This validation script is designed to be run in a separate process.

from hydra.utils import instantiate

Network = instantiate({"_target_": config.model_target, "_partial_": True})

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
# DEVICE = torch.device('cpu')
DEVICE = config.val_device
torch.set_num_threads(1)


def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    num_comm = 0
    if env_set[0].shape[0] == 40:
        max_episode_length = config.max_episode_length
    elif env_set[0].shape[0] == 80:
        max_episode_length = config.max_episode_length_80
    else:
        raise ValueError("Invalid map length")
    
    while not done and env.steps < max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)).to(DEVICE))
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    arrived_num = np.sum(np.all(env.agents_pos == env.goals_pos, axis=1)) 

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm, arrived_num


def test_model(checkpoint_path, test_set: Tuple = tuple(config.val_test_set)):
    '''
    validate model in 'saved_models' folder
    validation set path is './valid_set'
    Each validation set has 16 cases

    '''

    # Context for faster inference
    # https://pytorch.org/docs/stable/generated/torch.inference_mode.html
    with torch.inference_mode():
        network = Network()
        network.eval()
        network.to(DEVICE)

        pool = mp.Pool(config.val_pool_parallel_cores)

        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print(f'----------validate model {checkpoint_path}----------')

        success_rate, average_step, communication_times, average_arrived_agents, times = [], [], [], [], []
        case_fmt = []
        for case in test_set:
            init_time = time.time()
            print(f"valid set: {case[0]} length {case[1]} agents {case[2]} density")
            with open('./valid_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success, steps, num_comm, arrived = zip(*ret)


            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("average step: {}".format(sum(steps)/len(steps)))
            # print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print(f"average arrived agents : {sum(arrived)/len(arrived)} ")
            print()

            fmt = f'Agents#{case[1]}'

            case_fmt.append(fmt)
            success_rate.append(sum(success)/len(success)*100)
            average_step.append(sum(steps)/len(steps))
            communication_times.append(sum(num_comm)/len(num_comm))
            average_arrived_agents.append(sum(arrived)/len(arrived))
            times.append(time.time()-init_time)
        return success_rate, average_step, communication_times, average_arrived_agents, case_fmt, times    
                


def main(checkpoint_path: str, step: int = 0, run_id: str = 'test', name: str = config.name):

    # Model file exists, proceed to test
    print(f"Validating...")
    a, b, c, d, e, times = test_model(checkpoint_path)
    
    # Log the results
    if config.use_wandb:
        # Initialize wandb
        # Note: run_id is used to resume a run
        wandb.init(project=config.project, name=name+"-val", id=run_id+"-val", config=dict(config), resume="allow")
        for i in range(len(a)):
            wandb.log({f'val_success_rate/{e[i]}': a[i], 
                        f'val_average_step/{e[i]}': b[i], 
                        f'val_communication_times/{e[i]}': c[i], 
                        f'val_average_arrived_agents/{e[i]}': d[i], 
                        f"val_time/{e[i]}": times[i]},
                        step=step)
        
    # wandb.finish()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='saved_models')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--run_id', type=str, default='test')
    parser.add_argument('--name', type=str, default='test')

    args = parser.parse_args()

    main(args.checkpoint_path, args.step, args.run_id, args.name)