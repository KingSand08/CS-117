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
dl_test_accuracy, dl_test_loss = run_dl(dl_model, 0.00018690285734990255, 32, 5)
print(f'Test accuracy: {dl_test_accuracy:.4f}, Test loss: {dl_test_loss:.4f}')
dl_model.save("models/dl_model.keras")

print("------------------------------\nK-NN MODEL:")
knn_model = knn_builder(23, 'distance', 2)
knn_test_accuracy = run_knn(knn_model)
print(f'Test accuracy: {knn_test_accuracy:.4f}')
joblib.dump(knn_model, "models/knn_model.pkl")

print("------------------------------\nENSEMBLE VOTING (DL + K-NN):")
# voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0.001, 23, 'distance', 2, 0)
voting_model = build_ensemble_voting((35,51), 'tanh', 'adam', 127, 64, 3, 0.257857374573552, 0.0804721184915089, 23,'distance', 2, 0)
voting_accuracy = run_ensemble_voting(voting_model)
print(f'Test accuracy: {voting_accuracy:.4f}')
joblib.dump(voting_model, "models/voting_model.pkl")

print("------------------------------\nENSEMBLE BAGGING (DL):")
# dl_bagging_accuracy = run_ensemble_bagging_dl(1, (32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0, 0.001)
dl_bagging_model = build_ensemble_bagging_dl(32, (82,52), 'tanh', 'adam', 214, 32, 4, 0.14997869031270125, 0, 0.049818368013431334)
dl_bagging_accuracy = run_ensemble_bagging_dl(dl_bagging_model)
print(f'Test accuracy: {dl_bagging_accuracy:.4f}')
joblib.dump(dl_bagging_model, "models/dl_bagging_model.pkl")

print("------------------------------\nENSEMBLE VOTING (K-NN):")
# knn_bagging_model = build_ensemble_bagging_knn(100, 23, 'distance', 2)
knn_bagging_model = build_ensemble_bagging_knn(167, 23,'distance', 2)
knn_bagging_accuracy = run_ensemble_bagging_knn(knn_bagging_model)
print(f'Test accuracy: {knn_bagging_accuracy:.4f}')
joblib.dump(knn_bagging_model, "models/knn_bagging_model.pkl")
