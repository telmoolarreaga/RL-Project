import numpy as np
import random

def train_sarsa(env, alpha, gamma, epsilon, episodes, epsilon_decay):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        def choose_action(s):
            if random.random() < epsilon:
                return env.action_space.sample()
            return np.argmax(Q[s, :])

        action = choose_action(state)

        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(next_state)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_reward += reward

        epsilon *= epsilon_decay
        rewards.append(total_reward)

    return np.mean(rewards[-100:]), Q
