import numpy as np
from DL_training_final_project import dl_builder, run_dl
from k_NN_training_final_project import knn_builder, run_knn
from format_data import format_data as data
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load data
train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

# Soft voting: average predicted probabilities
dl_probs = run_dl(dl_builder())[0]       # shape: (50, 2)
knn_probs = run_knn(knn_builder())[0]     

avg_probs = (dl_probs + knn_probs) / 2.0
final_preds = np.argmax(avg_probs, axis=1)

# Accuracy
accuracy = accuracy_score(test_targets, final_preds)
print("Soft Voting Accuracy:", accuracy)
