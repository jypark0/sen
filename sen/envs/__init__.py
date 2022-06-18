from gym.envs.registration import register

register(
    "BlockPushing-v0",
    entry_point="sen.envs.block_pushing:BlockPushing",
)

register(
    "ReacherFixedGoal-v0",
    entry_point="sen.envs.reacher:ReacherFixedGoal",
)
