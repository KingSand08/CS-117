from Ensemble_Learning import run_ensemble_voting, run_ensemble_bagging_dl, run_ensemble_bagging_knn
from k_NN_training_final_project import knn_builder, run_knn
from DL_training_final_project import dl_builder, run_dl

print("DEEP LEARNING MODEL:")
dl_test_accuracy, dl_test_loss = run_dl(dl_builder())
print(f'Test accuracy: {dl_test_accuracy:.4f}, Test loss: {dl_test_loss:.4f}')

print("------------------------------\nK-NN MODEL:")
knn_test_accuracy = run_knn(knn_builder())
print(f'Test accuracy: {knn_test_accuracy:.4f}')

print("------------------------------\nENSEMBLE VOTING (DL + K-NN):")
voting_accuracy = run_ensemble_voting((32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0, 0.001)
print(f'Test accuracy: {voting_accuracy:.4f}')
print("------------------------------\nENSEMBLE BAGGING (DL):")
dl_bagging_accuracy = run_ensemble_bagging_dl(1, (32,32), 'relu', 'adam', 200, 100, 3, 0.5, 0, 0.001)
print(f'Test accuracy: {dl_bagging_accuracy:.4f}')
print("------------------------------\nENSEMBLE VOTING (K-NN):")
knn_bagging_accuracy = run_ensemble_bagging_knn(100)
print(f'Test accuracy: {knn_bagging_accuracy:.4f}')