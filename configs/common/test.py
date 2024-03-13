test_timeout = int(
    60 * 5
)  # seconds (integer!) after which the test will be stopped (no success)

use_wandb_test = False
test_seed = 0
num_test_cases = 100
test_folder = "./test_set"
test_data_mode = "normal"  # also available: SACHA
use_stepinfer_method = False
astar_type = 2  # 1: All agents are treated as obstacles, 2: Treat Finished agents as obstacles, 3: Not regarding any agent as obstacles, None: Not use astar

# AEP
use_aep = False
aep_distance_threshold = None
aep_use_q_value_only = False
aep_astar_type = 2

# Ensemble (list of tuples made of 3 elements)
test_env_settings = (
    (40, 4, 0.3),
    (40, 8, 0.3),
    (40, 16, 0.3),
    (40, 32, 0.3),
    (40, 64, 0.3),
    (80, 4, 0.3),
    (80, 8, 0.3),
    (80, 16, 0.3),
    (80, 32, 0.3),
    (80, 64, 0.3),
    (80, 128, 0.3),
)  # map length, number of agents, density
ensemble = None

# Other defaults
max_episode_length = 256
max_episode_length_80 = 386
max_episode_warehouse = 512
save_positions = False  # will trajectories (=positions in time) for each agent
save_map_config = False  # will save maps and config in the results
