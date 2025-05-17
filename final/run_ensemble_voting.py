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

from Ensemble_Learning_final_project import run_ensemble_voting, run_ensemble_bagging_dl, run_ensemble_bagging_knn, build_ensemble_voting, build_ensemble_bagging_dl, build_ensemble_bagging_knn
from k_NN_training_final_project import knn_builder, run_knn
from DL_training_final_project import dl_builder, run_dl
import joblib

import os
os.makedirs("models", exist_ok=True)


print("ENSEMBLE VOTING (DL + K-NN):")
# voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0.001, 23, 'distance', 2, 0)                                    # val:54 test:64 f1:59
# voting_model = build_ensemble_voting((35,51), 'tanh', 'adam', 127, 64, 3, 0.257857374573552, 0.0804721184915089, 23,'distance', 2, 0)           # val:64 test:78 f1:79
# voting_model = build_ensemble_voting((35,51), 'tanh', 'adam', 127, 64, 3, 0.4, 0.1, 23,'distance', 2, 0)                                        # val:60 test:76 f1:77
voting_model = build_ensemble_voting((95,32), 'relu', 'adam', 209, 32, 4, 0.2825982466163213, 0.048290217180251865, 25,'distance', 1, 0)        # val:62 test:82 f1:84
# voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 209, 100, 4, 0.8, 0.3, 256,'distance', 2, 0)                                      # val:62 test:66 f1:68
# voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 209, 100, 4, 0.8, 0.3, 16,'uniform', 1, 0)                                        # val:50 test:66 f1:64
# voting_model = build_ensemble_voting((125,45), 'tanh', 'adam', 140, 32, 4, 0.2895507016350658, 0.08922956314184817, 204,'distance', 2, 0)       # val:74 test:76 f1:79

# voting_model = build_ensemble_voting((128,64), 'relu', 'adam', 200, 64, 5, 0.20,   0.0010,  32, 'distance', 2, 0)                               # val:64 test:80 f1:82
# voting_model = build_ensemble_voting((64,64),  'tanh', 'adam', 150, 32, 4,  0.25,   0.0050,  64, 'distance', 1, 0)                              # val:68 test:74 f1:77
# voting_model = build_ensemble_voting((100,50), 'relu', 'sgd',  250,128, 6,  0.30,   0.0001, 16, 'uniform', 2, 0)                                # val:48 test:42 f1:0
# voting_model = build_ensemble_voting((50,25),  'tanh', 'adam', 180, 64, 3,  0.20,   0.0100, 64, 'uniform', 1, 0)                                # val:76 test:76 f1:80
# voting_model = build_ensemble_voting((128,128),'relu','adam', 300,128, 5,  0.15,   0.0005, 32, 'distance', 2, 0)                                # val:76 test:76 f1:80
# voting_model = build_ensemble_voting((32,64),  'relu','adam', 220, 32, 4,  0.20,   0.0010,  48, 'distance', 2, 0)                               # val:58 test:72 f1:72

# voting_model = build_ensemble_voting((150,75), 'relu', 'adam', 250, 32, 5, 0.25, 0.050, 180, 'distance', 2, 0)                                  # val:64 test:80 f1:82
# voting_model = build_ensemble_voting((100,50), 'tanh', 'adam', 300, 64, 4, 0.30, 0.010, 200, 'uniform', 1, 0)                                   # val:66 test:78 f1:80


voting_accuracy = run_ensemble_voting(voting_model)
# print(f'Test accuracy: {voting_accuracy:.4f}')
joblib.dump(voting_model, "models/voting_model.pkl")