import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_settings import seed, np, tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import optuna
from Ensemble_Learning_final_project import run_ensemble_bagging_knn


def bagging_knn_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    return run_ensemble_bagging_knn(estimator_num=n_estimators)

def run_study(n_trials=50):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(bagging_knn_objective, n_trials=n_trials)
    
    print("Best trial:")
    print(f"   Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    return study