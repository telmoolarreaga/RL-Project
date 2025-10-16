import numpy as np
import random

def train_expected_sarsa(env, alpha, gamma, epsilon, episodes, epsilon_decay):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, truncated, _ = env.step(action)

            # Expected SARSA update
            pi_s = np.ones(n_actions) * (epsilon / n_actions)
            pi_s[np.argmax(Q[next_state, :])] += 1 - epsilon
            expected_q = np.dot(Q[next_state, :], pi_s)
            Q[state, action] += alpha * (reward + gamma * expected_q - Q[state, action])

            state = next_state
            total_reward += reward

        epsilon *= epsilon_decay
        rewards.append(total_reward)

    return np.mean(rewards[-100:]), Q
