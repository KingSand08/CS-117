seed = 42

import random
random.seed(seed)

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['PYTHONHASHSEED'] = str(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

from Ensemble_Learning_final_project import run_ensemble_bagging_knn, build_ensemble_bagging_knn
import joblib

import os
os.makedirs("models", exist_ok=True)


print("ENSEMBLE VOTING (K-NN):")

# knn_bagging_model = build_ensemble_bagging_knn(100, 23, 'distance', 2)    # val:60 test:80 f1:82.76
# knn_bagging_model = build_ensemble_bagging_knn(167, 23, 'distance', 2)    # val:60 test:80 f1:82.76
# knn_bagging_model = build_ensemble_bagging_knn(130, 213, 'uniform', 2)    # val:72 test:84 f1:87.5
# knn_bagging_model = build_ensemble_bagging_knn(149, 213, 'uniform', 2)    # val:72 test:84 f1:87.5

# knn_bagging_model = build_ensemble_bagging_knn(10, 32, 'uniform', 2)      # val:64 test:76 f1:79.31
# knn_bagging_model = build_ensemble_bagging_knn(20, 40, 'uniform', 2)      # val:74 test:78 f1:81.97
# knn_bagging_model = build_ensemble_bagging_knn(50, 48, 'uniform', 2)      # val:74 test:80 f1:83.33
# knn_bagging_model = build_ensemble_bagging_knn(100, 56, 'uniform', 2)     # val:72 test:78 f1:83.08
# knn_bagging_model = build_ensemble_bagging_knn(150, 64, 'uniform', 2)     # val:72 test:80 f1:83.87
# knn_bagging_model = build_ensemble_bagging_knn(200, 72, 'uniform', 2)     # val:70 test:82 f1:85.71
# knn_bagging_model = build_ensemble_bagging_knn(250, 80, 'uniform', 2)     # val:72 test:82 f1:85.71
# knn_bagging_model = build_ensemble_bagging_knn(300, 70, 'uniform', 2)     # val:70 test:82 f1:85.71
# knn_bagging_model = build_ensemble_bagging_knn(350, 85, 'distance', 1)    # val:70 test:74 f1:79.37

# knn_bagging_model = build_ensemble_bagging_knn(100, 213, 'uniform', 2)    # val:72 test:82 f1:86.15
# knn_bagging_model = build_ensemble_bagging_knn(130, 213, 'uniform', 1)    # val:70 test:80 f1:84.38
# knn_bagging_model = build_ensemble_bagging_knn(130, 213, 'distance', 1)   # val:68 test:82 f1:86.15
# knn_bagging_model = build_ensemble_bagging_knn(130, 213, 'distance', 2)   # val:72 test:82 f1:86.15
# knn_bagging_model = build_ensemble_bagging_knn(100, 128, 'distance', 2)   # val:76 test:82 f1:86.15
# knn_bagging_model = build_ensemble_bagging_knn(100, 64, 'distance', 2)    # val:68 test:78 f1:81.97
# knn_bagging_model = build_ensemble_bagging_knn(100, 64, 'uniform', 2)     # val:72 test:80 f1:83.87
# knn_bagging_model = build_ensemble_bagging_knn(100, 128, 'uniform', 2)    # val:72 test:80 f1:84.85


knn_bagging_accuracy = run_ensemble_bagging_knn(knn_bagging_model)
# print(f'Test accuracy: {knn_bagging_accuracy:.4f}')
joblib.dump(knn_bagging_model, "models/knn_bagging_model.pkl")
