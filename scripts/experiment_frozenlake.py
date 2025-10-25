import os
import random
import json
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ------------------------------
# Imports from your scripts
# ------------------------------
from scripts.sarsa import train_sarsa
from scripts.expected_sarsa import train_expected_sarsa
from scripts.q_learning import train_q_learning
from scripts.utils import evaluate_policy  # evaluation helper

# ------------------------------
# Objective function for Optuna
# ------------------------------
def objective(trial):
    # Choose algorithm
    algo_name = trial.suggest_categorical("algo_name", ["sarsa", "esarsa", "qlearning"])
    print(f"Starting trial {trial.number+1} with algorithm: {algo_name}")

    # Hyperparameters
    alpha = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.9999)
    n_steps = 10_000  # total training steps per trial

    # Create environment (16x16 FrozenLake)
    map = generate_random_map(size=16, p=0.95)
    env = FrozenLakeEnv(desc=map, is_slippery=True)
    env = TimeLimit(env, max_episode_steps=400)

    # Train selected algorithm
    if algo_name == "sarsa":
        Q = train_sarsa(env, alpha, gamma, epsilon, n_steps, epsilon_decay)
    elif algo_name == "esarsa":
        Q = train_expected_sarsa(env, alpha, gamma, epsilon, n_steps, epsilon_decay)
    elif algo_name == "qlearning":
        Q = train_q_learning(env, alpha, gamma, epsilon, n_steps, epsilon_decay)

    # Evaluate learned policy
    avg_reward, avg_steps = evaluate_policy(env, Q, policy=None, n_episodes=100)
    env.close()
    return avg_reward

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    seed = 42
    random.seed(seed)

    # Prepare directories
    os.makedirs("optuna", exist_ok=True)

    # Optuna study setup
    storage_file = "sqlite:///optuna/optuna_frozenlake.db"
    study_name = "frozenlake16x16"
    full_study_dir_path = f"optuna/{study_name}"
    os.makedirs(full_study_dir_path, exist_ok=True)

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=study_name,
        storage=storage_file,
        load_if_exists=True,
    )

    n_trials = 150  # 50 per algorithm
    print(f"Running {n_trials} trials for {study_name}...")

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_trial = study.best_trial
    print("\nBest trial parameters:")
    print(json.dumps(best_trial.params, indent=4))
    print("Best average reward:", best_trial.value)

    # Save best hyperparameters
    os.makedirs(full_study_dir_path, exist_ok=True)
    with open(f"{full_study_dir_path}/best_trial.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)

    # Generate Optuna visualizations
    optuna.visualization.plot_optimization_history(study).write_html(
        f"{full_study_dir_path}/optimization_history.html"
    )
    optuna.visualization.plot_contour(study).write_html(
        f"{full_study_dir_path}/contour.html"
    )
    optuna.visualization.plot_slice(study).write_html(
        f"{full_study_dir_path}/slice.html"
    )
    optuna.visualization.plot_param_importances(study).write_html(
        f"{full_study_dir_path}/param_importances.html"
    )

    print(f"Study completed. Results stored in {full_study_dir_path}")
