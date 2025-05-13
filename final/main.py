from Ensemble_Learning_final_project import run_ensemble_voting, run_ensemble_bagging_dl, run_ensemble_bagging_knn, build_ensemble_voting, build_ensemble_bagging_dl, build_ensemble_bagging_knn
from k_NN_training_final_project import knn_builder, run_knn
from DL_training_final_project import dl_builder, run_dl
import joblib

import os
os.makedirs("models", exist_ok=True)


print("DEEP LEARNING MODEL:")
dl_model = dl_builder()
dl_test_accuracy, dl_test_loss = run_dl(dl_model)
print(f'Test accuracy: {dl_test_accuracy:.4f}, Test loss: {dl_test_loss:.4f}')
dl_model.save("models/dl_model.keras")

print("------------------------------\nK-NN MODEL:")
knn_model = knn_builder()
knn_test_accuracy = run_knn(knn_model)
print(f'Test accuracy: {knn_test_accuracy:.4f}')
joblib.dump(knn_model, "models/knn_model.pkl")

print("------------------------------\nENSEMBLE VOTING (DL + K-NN):")
voting_model = build_ensemble_voting((32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0, 0.001)
voting_accuracy = run_ensemble_voting(voting_model)
print(f'Test accuracy: {voting_accuracy:.4f}')
joblib.dump(voting_model, "models/voting_model.pkl")

print("------------------------------\nENSEMBLE BAGGING (DL):")
# dl_bagging_accuracy = run_ensemble_bagging_dl(1, (32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0, 0.001)
#   Params: {'layer_1': 82, 'layer_2': 52, 'activation': 'tanh', 'solver': 'adam', 'max_epochs': 214, 'batch_size': 32, 'patience': 4, 'val_fraction': 0.14997869031270125, 'alpha': 0.049818368013431334, 'n_estimators': 32}
dl_bagging_model = build_ensemble_bagging_dl(32, (82,52), 'tanh', 'adam', 214, 32, 4, 0.14997869031270125, 0, 0.049818368013431334)
dl_bagging_accuracy = run_ensemble_bagging_dl(dl_bagging_model)
print(f'Test accuracy: {dl_bagging_accuracy:.4f}')
joblib.dump(dl_bagging_model, "models/dl_bagging_model.pkl")

print("------------------------------\nENSEMBLE VOTING (K-NN):")
# knn_bagging_model = build_ensemble_bagging_knn(100)
# knn_bagging_accuracy = run_ensemble_bagging_knn(knn_bagging_model)
knn_bagging_model = build_ensemble_bagging_knn(167)
knn_bagging_accuracy = run_ensemble_bagging_knn(knn_bagging_model)
print(f'Test accuracy: {knn_bagging_accuracy:.4f}')
joblib.dump(knn_bagging_model, "models/knn_bagging_model.pkl")
