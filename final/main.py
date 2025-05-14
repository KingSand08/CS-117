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


print("DEEP LEARNING MODEL:")
# dl_model = dl_builder(32, 32, 0.3, 0.3)
# dl_test_accuracy, dl_test_loss = run_dl(dl_model, 0.001, 100, 5)
dl_model = dl_builder(116, 48, 0.3232858530601885, 0.330884719103952)
dl_test_accuracy, dl_test_loss = run_dl(dl_model, 0.00018690285734990255, 32, 5)    #51% #!OPTUNA APPROVED? (80%)
print(f'Test accuracy: {dl_test_accuracy:.4f}, Test loss: {dl_test_loss:.4f}')
dl_model.save("models/dl_model.keras")

print("------------------------------\nK-NN MODEL:")
knn_model = knn_builder(23, 'distance', 2)      #68% #!OPTUNA APPROVED? (86%?)
# knn_model = knn_builder(175, 'distance', 2)     #70%
# knn_model = knn_builder(165, 'distance', 2)     #70%
# knn_model = knn_builder(165, 'uniform', 2)      #72%
# knn_model = knn_builder(200, 'uniform', 2)      #72%
# knn_model = knn_builder(170, 'uniform', 2)      #74%
# knn_model = knn_builder(168, 'uniform', 2)      #74%✅
knn_test_accuracy = run_knn(knn_model)
print(f'Test accuracy: {knn_test_accuracy:.4f}')
joblib.dump(knn_model, "models/knn_model.pkl")

print("------------------------------\nENSEMBLE VOTING (DL + K-NN):")
# voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0.001, 23, 'distance', 2, 0)
voting_model = build_ensemble_voting((35,51), 'tanh', 'adam', 127, 64, 3, 0.257857374573552, 0.0804721184915089, 23,'distance', 2, 0)   #70% #!OPTUNA APPROVED? (86%?)
# voting_model = build_ensemble_voting((35,51), 'tanh', 'adam', 127, 64, 3, 0.4, 0.1, 23,'distance', 2, 0)                                #70
voting_accuracy = run_ensemble_voting(voting_model)
print(f'Test accuracy: {voting_accuracy:.4f}')
joblib.dump(voting_model, "models/voting_model.pkl")

print("------------------------------\nENSEMBLE BAGGING (DL):")
# dl_bagging_model = build_ensemble_bagging_dl(1, (32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0.001, 0)                                        #76%
# dl_bagging_model = build_ensemble_bagging_dl(32, (82,52), 'tanh', 'adam', 214, 32, 4, 0.14997869031270125, 0.049818368013431334, 0)         #78%
# dl_bagging_model = build_ensemble_bagging_dl(47, (111,106), 'tanh', 'adam', 125, 100, 8, 0.20317385092061593, 0.09719773091207945, 0)       #78%
# dl_bagging_model = build_ensemble_bagging_dl(47, (150,106), 'tanh', 'adam', 125, 100, 8, 0.20317385092061593, 0.09719773091207945, 0)       #78%
# dl_bagging_model = build_ensemble_bagging_dl(139, (67,18), 'relu', 'adam', 298, 32, 3, 0.18877591136046146, 0.03928683897516851, 0)         #82%✅
dl_bagging_model = build_ensemble_bagging_dl(155, (18,89), 'relu', 'adam', 129, 64, 5, 0.21328077721644959, 0.07750785574909076, 0)         #80% #!OPTUNA APPROVED? (90%?)
dl_bagging_accuracy = run_ensemble_bagging_dl(dl_bagging_model)
print(f'Test accuracy: {dl_bagging_accuracy:.4f}')
joblib.dump(dl_bagging_model, "models/dl_bagging_model.pkl")

print("------------------------------\nENSEMBLE VOTING (K-NN):")
# knn_bagging_model = build_ensemble_bagging_knn(100, 23, 'distance', 2)
knn_bagging_model = build_ensemble_bagging_knn(167, 23,'distance', 2)
knn_bagging_accuracy = run_ensemble_bagging_knn(knn_bagging_model)  #78% #!OPTUNA APPROVED? (84%?)
print(f'Test accuracy: {knn_bagging_accuracy:.4f}')
joblib.dump(knn_bagging_model, "models/knn_bagging_model.pkl")
