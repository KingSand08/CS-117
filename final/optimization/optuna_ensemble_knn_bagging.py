import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_settings import seed, np, tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from Ensemble_Learning_final_project import build_ensemble_bagging_knn, run_ensemble_bagging_knn


def bagging_knn_objective(trial):
    
    # Search space
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    n_neighbors = trial.suggest_int('n_neighbors', 1, 256)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)  # 1 = Manhattan, 2 = Euclidean

        
    return run_ensemble_bagging_knn(build_ensemble_bagging_knn(
        estimator_num=n_estimators, 
        n_neighbors_size=n_neighbors, 
        weights_size=weights,
        p_size=p
    ))