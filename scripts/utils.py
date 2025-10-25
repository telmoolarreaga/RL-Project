import numpy as np

def evaluate_policy(env, Q, policy=None, n_episodes=100):
    """
    Evaluate a Q-table policy in the environment.

    Args:
        env: Gymnasium environment
        Q: Q-table (numpy array of shape [n_states, n_actions])
        policy: Optional function mapping state -> action. If None, greedy policy is used.
        n_episodes: Number of episodes to evaluate

    Returns:
        avg_reward: Average reward over episodes
        avg_steps: Average steps per episode
    """
    total_rewards = []
    total_steps = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done:
            if policy is not None:
                action = policy(state)
            else:
                action = np.argmax(Q[state, :])  # greedy from Q-table
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        total_rewards.append(total_reward)
        total_steps.append(steps)

    return np.mean(total_rewards), np.mean(total_steps)
