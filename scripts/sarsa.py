import numpy as np
import random

def train_sarsa(env, alpha, gamma, epsilon, n_steps, epsilon_decay=1.0):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    steps_done = 0

    while steps_done < n_steps:
        state, _ = env.reset()
        done = False

        def choose_action(s):
            if random.random() < epsilon:
                return env.action_space.sample()
            return np.argmax(Q[s, :])

        action = choose_action(state)

        while not done and steps_done < n_steps:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(next_state)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            steps_done += 1

        epsilon *= epsilon_decay

    return Q
