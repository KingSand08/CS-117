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

from Ensemble_Learning_final_project import run_ensemble_bagging_dl, build_ensemble_bagging_dl
import joblib

import os
os.makedirs("models", exist_ok=True)

print("------------------------------\nENSEMBLE BAGGING (DL):")
# dl_bagging_model = build_ensemble_bagging_dl(1, (32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0.001, 0)                                        # val:74 test:76 f1:80
# dl_bagging_model = build_ensemble_bagging_dl(32, (82,52), 'tanh', 'adam', 214, 32, 4, 0.14997869031270125, 0.049818368013431334, 0)         # val:70 test:78 f1:81.97
# dl_bagging_model = build_ensemble_bagging_dl(47, (111,106), 'tanh', 'adam', 125, 100, 8, 0.20317385092061593, 0.09719773091207945, 0)       # val:72 test:78 f1:81.97
# dl_bagging_model = build_ensemble_bagging_dl(47, (150,106), 'tanh', 'adam', 125, 100, 8, 0.20317385092061593, 0.09719773091207945, 0)       # val:70 test:82 f1:85.25
# dl_bagging_model = build_ensemble_bagging_dl(139, (67,18), 'relu', 'adam', 298, 32, 3, 0.18877591136046146, 0.03928683897516851, 0)         # val:68 test:82 f1:85.25
# dl_bagging_model = build_ensemble_bagging_dl(8, (64,95), 'relu', 'adam', 400, 64, 35, 0.8, 0.2, 0)                                          # val:66 test:86 f1:88.14
# dl_bagging_model = build_ensemble_bagging_dl(155, (18,89), 'relu', 'adam', 129, 64, 5, 0.21328077721644959, 0.07750785574909076, 0)         # val:70 test:80 f1:83.87
# dl_bagging_model = build_ensemble_bagging_dl(8, (64,200,32), 'tanh', 'adam', 400, 64, 35, 0.8, 0.2, 0)                                      # val:72 test:82 f1:85.71

# dl_bagging_model = build_ensemble_bagging_dl(10, (128,64), 'relu', 'adam', 300, 32, 30, 0.70, 0.050, 0)                                     # val:70 test:84 f1:86.67
# dl_bagging_model = build_ensemble_bagging_dl(12, (64,128), 'relu', 'adam', 300, 64, 40, 0.60, 0.010, 0)                                     # val:68 test:80 f1:82.14
# dl_bagging_model = build_ensemble_bagging_dl(16, (100,50), 'tanh', 'adam', 250, 32, 20, 0.50, 0.100, 0)                                     # val:72 test:80 f1:83.33
# dl_bagging_model = build_ensemble_bagging_dl(20, (50,100), 'relu', 'adam', 350, 118, 30, 0.70, 0.010, 0)                                    # val:72 test:80 f1:82.76
# dl_bagging_model = build_ensemble_bagging_dl(24, (128,32), 'tanh', 'adam', 400, 64, 50, 0.80, 0.001, 0)                                     # val:72 test:80 f1:83.33
# dl_bagging_model = build_ensemble_bagging_dl(32, (64,64),  'relu', 'adam', 300, 16, 25, 0.60, 0.050, 0)                                     # val:72 test:82 f1:84.21
# dl_bagging_model = build_ensemble_bagging_dl( 8, (128,128),'relu', 'adam', 500, 64, 40, 0.70, 0.020, 0)                                     # val:72 test:80 f1:82.76


dl_bagging_accuracy = run_ensemble_bagging_dl(dl_bagging_model)
joblib.dump(dl_bagging_model, "models/dl_bagging_model.pkl")