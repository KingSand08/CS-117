import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from format_data import format_data as data

def knn_objective(trial):
    # Load data
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

    # Search space
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)  # 1 = Manhattan, 2 = Euclidean

    # Build model
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p
    )

    model.fit(train_inputs, train_targets)
    preds = model.predict(validation_inputs)
    accuracy = accuracy_score(validation_targets, preds)

    return accuracy  # Optuna will maximize this

def build_best_knn_model(best_params):
    return KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        p=best_params['p']
    )

def run_study(n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(knn_objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print(f"  Params: {trial.params}")
    
    return trial
