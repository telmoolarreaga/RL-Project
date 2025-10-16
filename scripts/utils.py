# scripts/utils.py
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
from gymnasium.wrappers import TimeLimit

def make_env(size=26, p=0.95, max_steps=800):
    """
    Creates a Frozen Lake environment of given size.
    
    Args:
        size (int): size of the lake (size x size)
        p (float): probability that a tile is frozen (not a hole)
        max_steps (int): maximum steps per episode

    Returns:
        env: FrozenLakeEnv wrapped with a TimeLimit
    """
    desc = generate_random_map(size=size, p=p)  # generate a random map
    env = FrozenLakeEnv(desc=desc, is_slippery=True)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env
