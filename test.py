'''create test set and test model'''
import os
import random
import pickle
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import Environment
from config import config
from utils import timeout
import time

from hydra.utils import instantiate


TIMEOUT_STEP_VAL = 1e9
@timeout(config.test_timeout, default_value=(False, TIMEOUT_STEP_VAL, 0, 0)) # default value for timeout is not success and large steps
def test_one_case(args, 
                  use_stepinfer_method: bool = True,
                    astar_type: int = 2,
                    active_agent_radius: int = 4,
                    use_aep: bool = True,
                    aep_astar_type: int = 2,
                  ):
    """
    Args:
        - args: (env_set, network)
        - use_stepinfer_method: bool. If True, use use step_infer method in Environment, else, do not use (`network.step` instead of `network.step_infer`)
        - astar_type: int. A* type. None (do not use astar), 1, 2, 3
        - active_agent_radius: int. Only if astar_type is not None. The radius of the active agent to be considered for astar
        - use_aep: bool. If True, use AEP, else, do not use
        
    Returns:
        - success: bool. If all agents arrived to their goals
        - steps: int. Number of steps
        - num_comms: int. Number of communications
        - arrived: int. Number of agents that arrived to their goals
    """    


    env_set, network, config = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    
    # Setting for inference techniques
    env.astar_type = astar_type
    env.active_agent_radius = active_agent_radius
    env.use_aep = use_aep
    env.aep_astar_type = aep_astar_type
    
    if use_stepinfer_method:
        obs, last_act, pos, no_agentnearby = env.observe_infer()
    else:
        obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    num_comm = 0
    # Let's make an array of recorded locations
    # location history: [num_agents, 4] # 4 previous steps
    loc_history = - np.ones((env.num_agents, 4, 2), dtype=int)
    # set history to -1 for first step to avoid conflict
    loc_history[:, :] = -42

    if env_set[0].shape[0] == 40:
        max_episode_length = config.max_episode_length
    elif env_set[0].shape[0] == 80:
        max_episode_length = config.max_episode_length_80
    else:
        # raise ValueError("Invalid map length")
        if config.data_mode == "normal":
            raise ValueError("Invalid map length")
        else:
            if env_set[0].shape[0] == 63:
                # as per SACHA's paper
                max_episode_length = config.max_episode_warehouse
            else:
                max_episode_length = config.max_episode_length
            # TODO: if there is "warehouse, then more"
                

    # device is taken from network parameters
    device = next(network.parameters()).device

    agent_positions = []
    while not done and env.steps < max_episode_length:
        actions, q_val, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(device), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(device), 
                                                    torch.as_tensor(pos.astype(int)).to(device))
        if use_stepinfer_method:
            (obs, last_act, pos, no_agentnearby), done, info, loc_history = env.step_infer(q_val, no_agentnearby, loc_history)
        else:
            (obs, last_act, pos), _, done, info = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)
        agent_positions.append(np.copy(env.agents_pos))

    arrived_num = np.sum(np.all(env.agents_pos == env.goals_pos, axis=1)) 
    agent_positions = np.array(agent_positions)
    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm, arrived_num, agent_positions
    

def test_instance(args):
    """
    Args:
        -(...) same as `test_one_case`
        - ensemble: list of tuples. Each tuple is a configuration for `test_one_case`. If not None, then use ensemble
    
    Returns:
        -(...) same as `test_one_case`
        - all_res: tuple. (successes, steps, num_comms, arrived). For logging for each ensemble case
    """

    test, network, config = args
    ensemble = config.ensemble

    # Take best result from ensemble
    if ensemble is not None:
        successes, steps, num_comms, arrived, positions = [], [], [], [], []
        for conf_ in ensemble:
            use_stepinfer_method, astar_type, active_agent_radius, use_aep, aep_astar_type = conf_
            su, st, nc, ar, ag_pos_ = test_one_case(args, 
                                                    use_stepinfer_method, astar_type, active_agent_radius, use_aep, aep_astar_type)
            if st == TIMEOUT_STEP_VAL:
                print(f"Timeout for ensemble {conf_}")
            successes.append(su)
            steps.append(st)
            num_comms.append(nc)
            arrived.append(ar)
            positions.append(ag_pos_)
        # take best result, namely the one with least steps (if not success, then most steps)
        successes = np.array(successes)
        steps = np.array(steps)
        num_comms = np.array(num_comms)
        arrived = np.array(arrived)
        # set steps of not success to a large number to avoid not success
        # make copy for function to work. steps will actually be from the best result
        temp_steps = steps.copy() # in case there is no succeess
        temp_steps[~successes] = 1e6
        idx = np.argmin(temp_steps)
        return successes[idx], steps[idx], num_comms[idx], arrived[idx], \
            successes, steps, num_comms, arrived, positions[idx], positions
    else:
        # Else, just run the test
        # return test_one_case(args, use_stepinfer_method, astar_type, active_agent_radius, use_aep, aep_astar_type)
        return test_one_case(args, config.use_stepinfer_method, config.astar_type, config.active_agent_radius, config.use_aep, config.aep_astar_type, config)

def load_data(case, config=config):
    # note: case is just a configuration
    if config.data_mode == "normal":
        map_name, num_agents, density = case
        print(f"test set: {map_name} length {num_agents} agents {density} density")
        with open('./{}/{}length_{}agents_{}density.pth'.format(config.test_folder, map_name, num_agents, density), 'rb') as f:
            tests = pickle.load(f)  
    elif config.data_mode == "movingai":
        # case: map_name, num_agents
        map_name, num_agents = case
        print(f"test set: {map_name} {num_agents} agents")
        with open(f'./{config.test_folder}/{map_name}_{num_agents}agents.pth', 'rb') as f:
        # with open('./{}/{}length_{}agents_{}density.pth'.format(config.test_folder, case[0], case[1], case[2]), 'rb') as f:
            def transform_list_element(x):
                # transfer to array when loading 
                map_ = np.array(x[0])
                agents_pos = np.array(x[1])
                goals_pos = np.array(x[2])
                return (map_, agents_pos, goals_pos)

            initial_tests = pickle.load(f)
            tests = [transform_list_element(x) for x in initial_tests]
    else: 
        raise ValueError(f"Invalid mode {config.data_mode}")
    return tests


def test_model(checkpoint_name: Union[int, str], config=config):
    '''
    test model in 'saved_models' folder
    '''
    with torch.inference_mode():
        Network = instantiate({"_target_": config.model_target, "_partial_": True})

        torch.manual_seed(config.test_seed)
        np.random.seed(config.test_seed)
        random.seed(config.test_seed)
        DEVICE = torch.device('cpu')
        torch.set_num_threads(1)

        network = Network()
        network.eval()
        network.to(DEVICE)


        pool = mp.Pool(mp.cpu_count()//2)
    

        state_dict = torch.load(os.path.join(config.save_path, str(checkpoint_name) + ".pth"), map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print(f'\n----------test model {config.name}@{checkpoint_name}----------')
        all_results = {}
        for case in config.test_env_settings:

            init_time = time.time()
                
            tests = load_data(case, config=config)

            tests = [(test, network, config) for test in tests]
            ret = tqdm(pool.imap(test_instance, tests), total=len(tests))

            if config.ensemble is not None:
                success, steps, num_comm, arrived, \
                    all_successes, all_steps, all_num_comms, all_arrived, best_pos, all_pos = zip(*ret)
            else:
                success, steps, num_comm, arrived, best_pos = zip(*ret)
                all_pos = best_pos
            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("average step: {}".format(sum(steps)/len(steps)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print(f"average arrived agents : {sum(arrived)/len(arrived)} ")
            print(f"Time taken for test set: {time.time()-init_time: .2f}")

            fmt = f'Agents#{case[1]}/'
                
            # Save results in pkl file
            if config.data_mode == "normal":
                subf_name = f"{case[0]}length_{case[1]}agents_{case[2]}density"
            else:
                subf_name = f"{case[0]}_{case[1]}agents"

            folder = f'./results/{subf_name}'
            os.makedirs(folder, exist_ok=True)
            # create dictionary of default results
            results = {
                "success": success,
                "steps": steps,
                "num_comm": num_comm,
                "arrived": arrived,
                # "best_pos": best_pos,
                # "pos": all_pos,
            }

            if config.save_positions:
                results["best_pos"] = best_pos
                results["pos"] = all_pos
            
            if config.save_map_config:
                results["map"] = [el[0] for el in tests] # 0 is the config
                
            # If ensemble, then save all results
            if config.ensemble is not None:
                # for conf_ in config.ensemble:
                all_successes = np.array(all_successes)
                all_steps = np.array(all_steps)
                all_num_comms = np.array(all_num_comms)
                all_arrived = np.array(all_arrived)
                for i, conf_ in enumerate(config.ensemble):
                    use_stepinfer, astar_type, active_agent_radius, use_aep, aep_astar_type = conf_
                    results[f"success_{use_stepinfer}_{astar_type}_{active_agent_radius}_{use_aep}_{aep_astar_type}"] = all_successes[:, i]
                    results[f"steps_{use_stepinfer}_{astar_type}_{active_agent_radius}_{use_aep}_{aep_astar_type}"] = all_steps[:, i]
                    results[f"num_comm_{use_stepinfer}_{astar_type}_{active_agent_radius}_{use_aep}_{aep_astar_type}"] = all_num_comms[:, i]
                    results[f"arrived_{use_stepinfer}_{astar_type}_{active_agent_radius}_{use_aep}_{aep_astar_type}"] = all_arrived[:, i]

            # Save
            print(f"Saving results in {folder}/{config.name}_{checkpoint_name}.pkl")
            with open(f'{folder}/{config.name}_{checkpoint_name}.pkl', 'wb') as f:
                pickle.dump(results, f)

            # return results
            all_results[case] = results

        return all_results
       


if __name__ == '__main__':


    # load trained model and reproduce results in paper
    # Set as environment variable "current_config" as the path to the config file
        
    test_model(65000)