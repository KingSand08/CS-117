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
from DL_training_final_project import data
import tensorflow as tf
import numpy as npz


train_inputs, train_targets, validation_inputs, validation_targets, _, _ = data()

def dl_objective(trial):
    # Suggest hyperparameters
    hidden_size1 = trial.suggest_int("hidden_size1", 16, 128)
    hidden_size2 = trial.suggest_int("hidden_size2", 16, 128)
    dropout_rate1 = trial.suggest_float("dropout_rate1", 0, 0.5)
    dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256])
    patience_size = trial.suggest_int("patience", 2, 8)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(train_inputs.shape[1],)),
        tf.keras.layers.Dense(hidden_size1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate1),
        tf.keras.layers.Dense(hidden_size2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_size,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        train_inputs,
        train_targets,
        epochs=200,
        batch_size=batch_size,
        validation_data=(validation_inputs, validation_targets),
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(validation_inputs, validation_targets, verbose=0)
    return val_accuracy