import random
import numpy as np
import copy

# config
import yaml
config = yaml.safe_load(open('./config.yaml', 'r'))
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']


# set random seed
print(f"Setting random seed to {config['seed']}")
random.seed(config['seed'])
np.random.seed(config['seed'])


def load_map(map_name):
    map_filename = config['map_files'][map_name]
    grid_map = []
    with open(map_filename) as fp:
        line = fp.readline()
        height = int(fp.readline().split(' ')[1])
        width = int(fp.readline().split(' ')[1])
        line = fp.readline()
        for _ in range(height):
            line = fp.readline()
            grid_map.append([])
            for cell in line:
                if cell == '@' or cell == 'T':
                    grid_map[-1].append(OBSTACLE)
                elif cell == '.':
                    grid_map[-1].append(FREE_SPACE)
    return grid_map


def generate_random_map(height, width, num_obstacles):
    grid_map = [[FREE_SPACE for _ in range(width)] for _ in range(height)]
    counter = 0
    while counter < num_obstacles:
        i = random.randint(0, height - 1)
        j = random.randint(0, width  - 1)
        if grid_map[i][j] == FREE_SPACE:
            grid_map[i][j] = OBSTACLE
            counter += 1
    return grid_map


def move(loc, d):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[d][0], loc[1] + directions[d][1]


def map_partition(grid_map):
    empty_spots = np.argwhere(np.array(grid_map)==FREE_SPACE).tolist()
    empty_spots = [tuple(pos) for pos in empty_spots]
    partitions = []
    while empty_spots:
        start_loc = empty_spots.pop()
        open_list = [start_loc]
        close_list = []
        while open_list:
            loc = open_list.pop(0)
            for d in range(4):
                child_loc = move(loc, d)
                if child_loc[0] < 0 or child_loc[0] >= len(grid_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(grid_map[0]):
                    continue
                if grid_map[child_loc[0]][child_loc[1]] == OBSTACLE:
                    continue
                if child_loc in empty_spots:
                    empty_spots.remove(child_loc)
                    open_list.append(child_loc)
            close_list.append(loc)
        partitions.append(close_list)
    return partitions


def generate_random_agents(grid_map, map_partitions, num_agents):
    starts, goals = [], []
    counter = 0
    partitions = copy.deepcopy(map_partitions)
    while counter < num_agents:
        partitions = [p for p in partitions if len(p) >= 2]
        partition_index = random.randint(0, len(partitions) - 1)
        si, sj = random.choice(partitions[partition_index])
        partitions[partition_index].remove((si, sj))
        gi, gj = random.choice(partitions[partition_index])
        partitions[partition_index].remove((gi, gj))
        starts.append((si, sj))
        goals.append((gi, gj))
        counter += 1
    return starts, goals