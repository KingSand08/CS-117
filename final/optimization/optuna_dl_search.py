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
from DL_training_final_project import dl_builder, run_dl
import tensorflow as tf
import numpy as npz


def dl_objective(trial):
    # Suggest hyperparameters
    hidden_size1 = trial.suggest_int("hidden_size1", 16, 128)
    hidden_size2 = trial.suggest_int("hidden_size2", 16, 128)
    dropout_rate1 = trial.suggest_float("dropout_rate1", 0, 0.5)
    dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256])
    patience_size = trial.suggest_int("patience", 2, 8)

    
    dl_model = dl_builder(hidden_size1, hidden_size2, dropout_rate1, dropout_rate2)
    dl_test_accuracy, dl_test_loss = run_dl(dl_model, learning_rate, batch_size, patience_size)     #51% #!OPTUNA APPROVED? (80%)


    return dl_test_accuracy


def dl_objective_1(trial):
    # Suggest hyperparameters
    hidden_size1 = trial.suggest_int("hidden_size1", 16, 128)
    hidden_size2 = trial.suggest_int("hidden_size2", 16, 128)
    hidden_size3 = trial.suggest_int("hidden_size3", 16, 128)
    dropout_rate1 = trial.suggest_float("dropout_rate1", 0, 0.5)
    dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.5)
    dropout_rate3 = trial.suggest_float("dropout_rate3", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256])
    patience_size = trial.suggest_int("patience", 2, 8)

    
    # dl_model = dl_builder(hidden_size1, hidden_size2, dropout_rate1, dropout_rate2)
    
    input_size = 21
    output_size = 2

    dl_model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_size,)),
        tf.keras.layers.Dense(hidden_size1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate1),
        tf.keras.layers.Dense(hidden_size2, activation='relu'), 
        tf.keras.layers.Dropout(dropout_rate2),
        tf.keras.layers.Dense(hidden_size3, activation='relu'), 
        tf.keras.layers.Dropout(dropout_rate3),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    
    dl_test_accuracy, dl_test_loss = run_dl(dl_model, learning_rate, batch_size, patience_size)     #51% #!OPTUNA APPROVED? (80%)


    return dl_test_accuracy