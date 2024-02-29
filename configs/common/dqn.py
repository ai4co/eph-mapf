# Optimizer
lr = 2e-4
lr_scheduler_milestones = [40000, 80000]
lr_scheduler_gamma = 0.5
weight_decay = 0 # as original, no weight decay

# basic training setting
num_actors = 16
log_interval = 10
training_steps = 150000
save_interval = 1000
gamma = 0.99
batch_size = 128
learning_starts = 50000
target_network_update_freq = 1750
max_episode_length = 256
buffer_capacity = 262144
chunk_capacity = 64
burn_in_steps = 20

actor_update_steps = 200

# gradient norm clipping
grad_norm_dqn=40

# n-step forward
forward_steps = 2

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# curriculum learning
init_env_settings = (1, 10)
max_num_agents = 16
max_map_lenght = 40
pass_rate = 0.9

# dqn network setting
cnn_channel = 128
hidden_dim = 256

# same as DHC if set to false
selective_comm = True
# only works if selective_comm set to false
max_comm_agents = 3

# curriculum learning
cl_history_size = 100

