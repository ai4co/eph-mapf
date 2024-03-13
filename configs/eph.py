import datetime

from common import *

# Wandb setting
use_wandb = False  # change to True if you want to use it!
project = "FinalTest"
name = "eph"
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Network setting
model_target = "src.models.eph.Network"


# For saving model
save_path = "./saved_models/eph_net"

##### Note #####
# If you want to override some parameters from the common/ folder, just
# override down here. For examples:
# active_agent_radius = 1


use_wandb_test = False
test_folder = "./test_set"

density = 0.3
test_env_settings = (
    (40, 4, density),
    (40, 8, density),
    (40, 16, density),
    (40, 32, density),
    (40, 64, density),
    (80, 4, density),
    (80, 8, density),
    (80, 16, density),
    (80, 32, density),
    (80, 64, density),
    (80, 128, density),
)  # map length, number of agents, density

########Part for Single Solver(Not ensemble)###########
# set ensemble = None if you want to use a single solver with the following parameters
use_stepinfer_method = True  # Priority 1, if False, the following parameters are not used
astar_type = 2
active_agent_radius = 4
# AEP
use_aep = True
aep_distance_threshold = None
aep_use_q_value_only = False
aep_astar_type = 2
#######################################################


""" Ensemble (list of tuples made of 5 elements)
use_stepinfer_method, astar_type, active_agent_radius, use_aep, aep_astar_type
- use_stepinfer_method: True or False
- astar_type: 1: All agents are treated as obstacles, 2: Treat Finished agents as obstacles, 3: Not regarding any agent as obstacles, None: Not use astar
- active_agent_radius: only if astar_type is not None. The radius of the active agent to be considered for astar. This parameter is only for hybrid guidance, NOT for AEP.
- use_aep: True or False
- aep_astar_type: 1: All agents are treated as obstacles, 2: Treat Finished agents as obstacles, 3: Not regarding any agent as obstacles, None: Not use astar
"""
ensemble = [
    (False, None, None, False, None),
    (True, None, 4, False, None),  # third value is not actually used
    (True, 1, 2, False, None),
    (True, 1, 3, False, None),
    (True, 1, 4, False, None),
    (True, 1, 5, False, None),
    (True, 2, 2, False, None),
    (True, 2, 3, False, None),
    (True, 2, 4, False, None),
    (True, 2, 5, False, None),
    (True, 1, 2, True, 1),
    (True, 1, 3, True, 1),
    (True, 1, 4, True, 1),
    (True, 1, 5, True, 1),
    (True, 2, 2, True, 2),
    (True, 2, 3, True, 2),
    (True, 2, 4, True, 2),
    (True, 2, 5, True, 2),
    (True, 3, 2, True, 3),
    (True, 3, 3, True, 3),
    (True, 3, 4, True, 3),
    (True, 3, 5, True, 3),
]
