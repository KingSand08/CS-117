from ml_settings import np, plt
from format_data import format_data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Returns a k-NN learning model
def knn_builder(n_neighbors_size, weights_size, p_size): 
   #! Create k-NN Model
   # Search space
   n_neighbors = n_neighbors_size
   weights = weights_size
   p = p_size # 1 = Manhattan, 2 = Euclidean

   # Build model
   knn_model = KNeighborsClassifier(
      n_neighbors=n_neighbors,
      weights=weights,
      p=p
   )
   
   return knn_model
    
def run_knn(knn_model):
   # Load Data
   train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

   #! Train Model
   history = knn_model.fit(train_inputs, train_targets)

   #! Evaluate on test set
   test_predicted_classes = knn_model.predict(test_inputs)
   accuracy = accuracy_score(test_targets, test_predicted_classes)

   cm = confusion_matrix(test_targets, test_predicted_classes).T
   class_names = ['No Stroke', 'Stroke']

   #? Print classification report
   # print("\nClassification Report (Test Set Only):")
   # print(classification_report(test_targets, test_predicted_classes, target_names=class_names))

   #? Validation Accuracy
   val_preds = knn_model.predict(validation_inputs)
   val_acc = accuracy_score(validation_targets, val_preds)
   print(f"Validation accuracy: {val_acc:.2f}")

   #? Test Accuracy
   test_preds = knn_model.predict(test_inputs)
   test_acc = accuracy_score(test_targets, test_preds)
   print(f"Test accuracy:       {test_acc:.2f}")
   
   #? F1 Stoke Score
   test_f1_stroke = f1_score(test_targets, test_preds, pos_label=1)
   print(f"F1-score (Stroke):   {test_f1_stroke:.2f}")

   return accuracy