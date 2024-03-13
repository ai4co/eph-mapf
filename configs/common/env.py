obs_radius = 4
active_agent_radius = 3
reward_fn = dict(
    move=-0.075, stay_on_goal=0, stay_off_goal=-0.075, collision=-0.5, finish=3
)

obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
action_dim = 5
