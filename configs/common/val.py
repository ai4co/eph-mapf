from .dqn import save_interval

use_validation = True
val_interval = save_interval
val_test_set = ((40,16,0.3),(40,32,0.3),(40,64,0.3))

# cpu cores
val_pool_parallel_cores = 16
val_device = "cpu"

