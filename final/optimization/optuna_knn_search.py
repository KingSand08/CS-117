import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from k_NN_training_final_project import run_knn, knn_builder
from ml_settings import seed, np, tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from format_data import format_data as data


def knn_objective(trial):
    # Load data
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

    # Search space
    n_neighbors = trial.suggest_int('n_neighbors', 1, 256)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)  # 1 = Manhattan, 2 = Euclidean

    # Build model
    # model = KNeighborsClassifier(
    #     n_neighbors=32,
    #     weights=weights,
    #     p=2
    # )

    # model.fit(train_inputs, train_targets)
    # preds = model.predict(validation_inputs)
    # accuracy = accuracy_score(validation_targets, preds)
    
    model = knn_builder(n_neighbors, weights, p)
    accuracy = run_knn(model)

    return accuracy