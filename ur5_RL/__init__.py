from gym.envs.registration import register

register(
    id='ur5_RL-v0',
    entry_point='ur5_RL.envs:ur5Env',
)

register(
    id='ur5_RL_objects-v0',
    entry_point='ur5_RL.envs:ur5Env_objects',
)


register(
    id='ur5_RL_relative-v0',
    entry_point='ur5_RL.envs:ur5Env_reacher_relative',
)

register(
    id='ur5_2Dreacher-v0',
    entry_point='ur5_RL.envs:ur5Env_2D',
)

register(
    id='ur5_2Dreacher_object-v0',
    entry_point='ur5_RL.envs:ur5Env_2D_objects',
)