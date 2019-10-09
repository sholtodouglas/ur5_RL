from gym.envs.registration import register

register(
    id='ur5_RL-v0',
    entry_point='ur5_RL.envs:ur5Env',
)

register(
    id='ur5_RL_objects-v0',
    entry_point='ur5_RL.envs:ur5Env_objects',
    max_episode_steps=50,
)
register(
    id='ur5_RL_tools-v0',
    entry_point='ur5_RL.envs:ur5Env_tools',
    max_episode_steps=80,
)


register(
    id='ur5_RL_2Dobjects-v0',
    entry_point='ur5_RL.envs:ur5Env_2D_objects',
    max_episode_steps=50,
)

register(
    id='ur5_RL_relative-v0',
    entry_point='ur5_RL.envs:ur5Env_reacher_relative',
    max_episode_steps=100,
)

register(
    id='ur5_2Dreacher-v0',
    entry_point='ur5_RL.envs:ur5Env_2D',
)


register(
    id='ur5_RL_pointmass-v0',
    entry_point='ur5_RL.envs:ur5Env_pointmasstest',
)

register(
    id='ur5_2Dpointmass_object-v0',
    entry_point='ur5_RL.envs:ur5Env_pointmasstest_object',
)
