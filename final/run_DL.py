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

from DL_training_final_project import dl_builder, run_dl, dl_builder_1

import os
os.makedirs("models", exist_ok=True)


# print("DEEP LEARNING MODEL (2 Hidden Layers):")
# dl_model = dl_builder(32, 32, 0.3, 0.3)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.001, 100, 5)                        # Test accuracy: 0.8400, Test loss: 0.4708

# dl_model = dl_builder(116, 48, 0.3232858530601885, 0.330884719103952)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.00018690285734990255, 32, 5)        # Test accuracy: 0.8000, Test loss: 0.5117

# dl_model = dl_builder(94, 17, 0.4543069003263215, 0.2769282562528372)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0004368035982131504, 64, 17)        # Test accuracy: 0.7800, Test loss: 0.4633

# dl_model = dl_builder(61, 91, 0.29143885436587874, 0.4445456645549816)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0012231985804781768, 64, 5)         # Test accuracy: 0.7800, Test loss: 0.4928

# dl_model = dl_builder(64, 95, 0.29143885436587874, 0.4445456645549816)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0012231985804781768, 64, 5)     # Test accuracy: 0.8600, Test loss: 0.4881

# dl_model = dl_builder(87, 41, 0.3711951154101628, 0.32052326258167985)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0011856664480114616, 64, 8)     # Test accuracy: 0.8200, Test loss: 0.4651

# dl_model = dl_builder(87, 41, 0.29143885436587874, 0.4445456645549816)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0011856664480114616, 64, 8)     # MIDDLE OF BOTH BESTS LOSS: Test accuracy: 0.8400, Test loss: 0.4797


# dl_model = dl_builder(87, 41, 0.29143885436587874, 0.4445456645549816)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0011856664480114616, 64, 5)     # MIDDLE OF BOTH BESTS LOSS: Test accuracy: 0.8400, Test loss: 0.4797 (NO DIFF ^^)

# dl_model = dl_builder(64, 95, 0.29143885436587874, 0.32052326258167985)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0011856664480114616, 64, 5)     # Test accuracy: 0.8600, Test loss: 0.4857

# dl_model = dl_builder(64, 95, 0.2914, 0.3205)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0011856, 64, 5)                 # Test accuracy: 0.8600, Test loss: 0.4607

# dl_model = dl_builder(64, 95, 0.29, 0.32)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.001, 100, 5)                    # Test accuracy: 0.8600, Test loss: 0.4607

# dl_model = dl_builder(64, 95, 0.29, 0.32)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.001, 100, 3)                    #? BEST VAL AND LOSS: Test accuracy: 0.8600, Test loss: 0.4607


print("DEEP LEARNING MODEL (3 Hidden Layers):")
# dl_model = dl_builder_1(32, 32, 32, 0.3, 0.3, 0.3)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.001, 100, 5)                    # Test accuracy: 0.7800, Test loss: 0.4856

# dl_model = dl_builder_1(92, 122, 35, 0.49747422290654897, 0.16157142118463466, 0.2601167246615356)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.00040492358684563644, 32, 7)    # Test accuracy: 0.7800, Test loss: 0.4958

# dl_model = dl_builder_1(80, 125, 25, 0.0, 0.0, 0.0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568738661706191, 16, 7)    # Test accuracy: 0.8000, Test loss: 0.5140

# dl_model = dl_builder_1(64, 128, 32, 0.0, 0.0, 0.0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568738661706191, 16, 7)    # Test accuracy: 0.6800, Test loss: 0.6257

# dl_model = dl_builder_1(80, 125, 25, 0, 0.01, 0.0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568738661706191, 16, 3)    # Test accuracy: 0.7600, Test loss: 0.5393

dl_model = dl_builder_1(80, 125, 25, 0, 0, 0.01)
dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568738661706191, 16, 3)    # Test accuracy: 0.7800, Test loss: 0.5181

# dl_model = dl_builder_1(80, 125, 25, 0.01, 0, 0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568738661706191, 16, 3)    # Test accuracy: 0.7800, Test loss: 0.5061

# dl_model = dl_builder_1(80, 125, 25, 0, 0, 0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0004, 16, 3)                   # Test accuracy: 0.7400, Test loss: 0.5039

# dl_model = dl_builder_1(80, 125, 25, 0, 0, 0)
# dl_test_accuracy, dl_test_loss, dl_f1_score = run_dl(dl_model, 0.0003568739, 16, 3)                # Test accuracy: 0.7400, Test loss: 0.5039

# print(f'Test accuracy: {dl_test_accuracy:.4f}, Test loss: {dl_test_loss:.4f}')
dl_model.save("models/dl_model.keras")