
import os
import random
import json
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_slice,
    plot_param_importances,
    plot_contour
)

from scripts.utils import make_env
from scripts.sarsa import train_sarsa
from scripts.expected_sarsa import train_expected_sarsa
from scripts.q_learning import train_q_learning


# Objective function
def objective(trial):
    # Select algorithm for this trial
    algo_name = trial.suggest_categorical("algo_name", ["sarsa", "expected_sarsa", "q_learning"])

    # Hyperparameter search space
    alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.01, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 1.0)
    episodes = 400  # Adjusted for 26x26 map

    env = make_env(size=26, p=0.95, max_steps=800)

    if algo_name == "sarsa":
        avg_reward, _ = train_sarsa(env, alpha, gamma, epsilon, episodes, epsilon_decay)
    elif algo_name == "expected_sarsa":
        avg_reward, _ = train_expected_sarsa(env, alpha, gamma, epsilon, episodes, epsilon_decay)
    else:
        avg_reward, _ = train_q_learning(env, alpha, gamma, epsilon, episodes, epsilon_decay)

    env.close()
    return avg_reward


# Main script
if __name__ == "__main__":
    random.seed(47)

    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Database & study config
    storage_file = "sqlite:///data/optuna_study.db"
    study_name = "FrozenLake_26x26_Study"

    sampler = TPESampler(seed=47)
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        study_name=study_name,
        storage=storage_file,
        load_if_exists=True
    )

    n_trials = 150  # 50 per algorithm
    print(f"Running Optuna study with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # Save best trial params to JSON
    best_trial = study.best_trial
    best_params = json.dumps(best_trial.params, indent=4)
    with open("results/best_hyperparameters.json", "w") as f:
        f.write(best_params)

    print("Best parameters:")
    print(best_params)

    # Save Optuna visualization plots
    print("Saving Optuna plots...")
    fig = plot_optimization_history(study)
    fig.write_html("results/plots/optimization_history.html")

    fig = plot_param_importances(study)
    fig.write_html("results/plots/param_importances.html")

    fig = plot_slice(study)
    fig.write_html("results/plots/slice.html")
    

    fig = plot_contour(study)
    fig.write_html("results/plots/contour.html")

