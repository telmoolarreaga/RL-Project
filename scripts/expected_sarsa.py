import numpy as np
import random

def train_expected_sarsa(env, alpha, gamma, epsilon, n_steps, epsilon_decay=1.0):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    steps_done = 0

    while steps_done < n_steps:
        state, _ = env.reset()
        done = False

        while not done and steps_done < n_steps:
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
            steps_done += 1

        epsilon *= epsilon_decay

    return Q
