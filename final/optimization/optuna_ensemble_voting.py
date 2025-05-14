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
from Ensemble_Learning_final_project import run_ensemble_voting


def voting_objective(trial):
    # Sample hyperparameters
    layer_sizes = tuple([trial.suggest_int("layer_1", 16, 128),
                         trial.suggest_int("layer_2", 0, 128)])
    # Remove 0-layer if needed
    layer_sizes = tuple([l for l in layer_sizes if l > 0])

    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    max_epochs = trial.suggest_int("max_epochs", 100, 300)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 100])
    patience = trial.suggest_int("patience", 3, 10)
    val_fraction = trial.suggest_float("val_fraction", 0.1, 0.3)
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1)

    return run_ensemble_voting(
        layerSizes=layer_sizes,
        act_func=activation,
        solver_func=solver,
        max_epochs=max_epochs,
        given_batch_size=batch_size,
        patience=patience,
        val_per=val_fraction,
        verboseness=0,
        early_stop_per=alpha
    )

def run_study(n_trials=50):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(voting_objective, n_trials=n_trials)
    
    print("Best trial:")
    print(f"   Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    return study