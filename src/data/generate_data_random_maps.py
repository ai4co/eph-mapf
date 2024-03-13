import copy
import pickle
import random

import numpy as np

from tqdm import tqdm

MAX_ITERATIONS = 1000
FREE_SPACE = 0
OBSTACLE = 1


def flood_fill(matrix, x, y, old_value, new_value):
    if x < 0 or x >= matrix.shape[0] or y < 0 or y >= matrix.shape[1]:
        return
    if matrix[x, y] != old_value:
        return
    matrix[x, y] = new_value
    flood_fill(matrix, x + 1, y, old_value, new_value)
    flood_fill(matrix, x - 1, y, old_value, new_value)
    flood_fill(matrix, x, y + 1, old_value, new_value)
    flood_fill(matrix, x, y - 1, old_value, new_value)


def generate_map(width, height, density, tolerance=0.005):

    iteration = 0

    while iteration < MAX_ITERATIONS:
        matrix = np.random.choice([0, 1], size=(width, height), p=[1 - density, density])

        filled_matrix = matrix.copy()
        flood_fill(filled_matrix, 0, 0, 0, 2)
        total_free_space = np.sum(filled_matrix == 2)
        total_space = width * height
        actual_density = 1 - total_free_space / total_space
        if abs(actual_density - density) < tolerance:
            filled_matrix[filled_matrix == 0] = 1
            filled_matrix[filled_matrix == 2] = 0
            return filled_matrix
        iteration += 1

    raise ValueError(
        f"Unable to generate a grid with the desired density of {density} after {MAX_ITERATIONS} iterations."
    )


def move(loc, d):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[d][0], loc[1] + directions[d][1]


def map_partition(grid_map):
    empty_spots = np.argwhere(np.array(grid_map) == FREE_SPACE).tolist()
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
                if (
                    child_loc[0] < 0
                    or child_loc[0] >= len(grid_map)
                    or child_loc[1] < 0
                    or child_loc[1] >= len(grid_map[0])
                ):
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
    # convert to numpy array
    starts = np.array(starts, dtype=int)
    goals = np.array(goals, dtype=int)

    return starts, goals


### Main function to generate data and save into pickle files
def main(height, width, density, num_agents, num_instances):
    instances = []

    print(f"Generating instances for {height}x{width} map with {num_agents} agents ...")
    for _ in tqdm(range(num_instances)):
        map_ = generate_map(height, width, density)
        map_partitions = map_partition(map_)
        starts, goals = generate_random_agents(map_, map_partitions, num_agents)
        instances.append((map_, starts, goals))
    file_name = f"./test_set/{height}length_{num_agents}agents_{density}density.pth"
    with open(file_name, "wb") as f:
        pickle.dump(instances, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random maps and agents for testing"
    )
    parser.add_argument("--width", type=int, default=40, help="Width of the map")
    parser.add_argument("--height", type=int, default=40, help="Height of the map")
    parser.add_argument("--density", type=float, default=0.3, help="Density of the map")
    parser.add_argument("--num_agents", type=int, default=4, help="Number of agents")
    parser.add_argument(
        "--num_instances", type=int, default=100, help="Number of instances"
    )
    args = parser.parse_args()

    main(args.height, args.width, args.density, args.num_agents, args.num_instances)
