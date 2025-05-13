import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from Ensemble_Learning import run_ensemble_bagging_knn


def bagging_knn_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    return run_ensemble_bagging_knn(estimator_num=n_estimators)

def run_study(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(bagging_knn_objective, n_trials=n_trials)
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    return study