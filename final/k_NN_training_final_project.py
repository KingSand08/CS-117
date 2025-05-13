from ml_settings import np, plt
from format_data import format_data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Returns a k-NN learning model
def knn_builder(n_neighbors_size, weights_size, p_size): 
   #! Create k-NN Model
   train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

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
    
# Run deep learning model given model from variable from dl_builder function
def run_knn(knn_model):
   # Load Data
   train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

   #! Train Model
   knn_model.fit(train_inputs, train_targets)

   #! Evaluate on test set
   test_predicted_classes = knn_model.predict(test_inputs)

   #? Evaluate Model
   cm = confusion_matrix(test_targets, test_predicted_classes).T
   class_names = ['No Stroke', 'Stroke']

   # Print classification report
   # print("\nClassification Report (Test Set Only):")
   # print(classification_report(test_targets, test_predicted_classes, target_names=class_names))

   predictions = knn_model.predict_proba(test_inputs)
   predicted_classes = np.argmax(predictions, axis=1)
   accuracy = accuracy_score(test_targets, predicted_classes)

   return accuracy
   # print(accuracy)

   # Plot confusion matrix for k-NN model
   # plt.figure(figsize=(6, 4))
   # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
   # plt.xlabel("Predicted")
   # plt.ylabel("Actual")
   # plt.title(f"k-NN Confusion Matrix (k={k})")
   # plt.tight_layout()
   # plt.show()