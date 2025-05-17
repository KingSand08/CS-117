import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_settings import seed

import optuna
from optuna.samplers import TPESampler
from optuna_dl_search import dl_objective as optuna_dl_test, dl_objective_1 as optuna_dl_test_extra_1_layer
from optuna_knn_search import knn_objective as optuna_knn_test
from optuna_ensemble_voting import voting_objective as optuna_voting_test
from optuna_ensemble_dl_bagging import bagging_dl_objective as optuna_dl_bagging_test
from optuna_ensemble_knn_bagging import bagging_knn_objective as optuna_knn_bagging_test

def run_study(objective, n_trials=50):
    # Run the study
    sampler = TPESampler(seed=42)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(study_name="k-NN Optimization Optuna Study", direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Print best result
    print("Best Trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params: ", study.best_params)
    
    return study


# run_study(optuna_dl_test, 10)
# run_study(optuna_dl_test_extra_1_layer, 800)

# run_study(optuna_knn_test, 20)

# run_study(optuna_voting_test, 500)

# run_study(optuna_dl_bagging_test, 1000)

run_study(optuna_knn_bagging_test, 1000)