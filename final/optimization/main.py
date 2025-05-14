import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_settings import seed

import optuna
from optuna_dl_search import dl_objective as optuna_dl_test
from optuna_knn_search import knn_objective as optuna_knn_test
from optuna_ensemble_voting import voting_objective as optuna_voting_test
from optuna_ensemble_dl_bagging import bagging_dl_objective as optuna_dl_bagging_test
from optuna_ensemble_knn_bagging import bagging_knn_objective as optuna_knn_bagging_test

def run_study(objective, n_trials=50):
    # Run the study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    # Print best result
    print("Best Trial:")
    print(f"  Value: {study.value:.4f}")
    print("  Params: ", study.best_params)
    
    return study


# run_study(optuna_dl_test, 1000)

# run_study(optuna_knn_test, 1000)

# run_study(optuna_voting_test, 1000)

run_study(optuna_dl_bagging_test, 1000)

# run_study(optuna_knn_bagging_test, 1000)