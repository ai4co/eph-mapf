import argparse
import pickle
from tqdm import tqdm

from .utils import load_map, map_partition, generate_random_agents

# config
import yaml

# If called from the command line, the following code will be executed
try:
    config = yaml.safe_load(open("./config.yaml", 'r'))
except:
    config = yaml.safe_load(open("./src/data/config.yaml", 'r'))


def generate_movingai(num_instances, test_env_settings=config['test_env_settings']):
    for map_name, num_agents in test_env_settings:
        instances = []
        grid_map = load_map(map_name)
        map_partitions = map_partition(grid_map)
        print(f"Generating instances for {map_name} with {num_agents} agents ...")
        for _ in tqdm(range(num_instances)):
            starts, goals = generate_random_agents(grid_map, map_partitions, num_agents)
            instances.append((grid_map, starts, goals))
        file_name = f"./test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'wb') as f:
            pickle.dump(instances, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", default=300, type=int)
    args = parser.parse_args()
    generate_movingai(args.num_instances)