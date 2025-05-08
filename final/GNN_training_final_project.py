from ml_settings import np, tf
import spektral as sp
from keras import Model
from sklearn.metrics.pairwise import cosine_similarity
from spektral.utils import normalized_adjacency
from scipy.sparse import csr_matrix
from spektral.layers import GCNConv
from spektral.utils import sp_matrix_to_sp_tensor
from format_data import format_data as data

# Extract the data from the formatting data function
train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

#! GNN Additional Preprocessing
# Use the full balanced + scaled input set (e.g., train_inputs + val_inputs + test_inputs)
node_features_all = np.concatenate([train_inputs, validation_inputs, test_inputs])  # Nodes of graph
labels_all = np.concatenate([train_targets, validation_targets, test_targets])      # Connections of graph

#? Cosine similarity-based adjacency matrix
adj_matrix_raw = cosine_similarity(node_features_all)      # Cosine similarity for all nodes
adj_matrix_raw[adj_matrix_raw < 0.9] = 0                   # Threshold for edges ("lack connection")
adj_matrix_raw[adj_matrix_raw >= 0.9] = 1                  # Binary adjacency matrix ("build connection")
adj_matrix_sparse = csr_matrix(adj_matrix_raw)             # Convert NumPy array to SciPy CSR matrix for necessary optimization to run
adj_matrix_norm = normalized_adjacency(adj_matrix_sparse)  # Normalize the binary adjacency matrix
similarity_adjacency_matrix = sp_matrix_to_sp_tensor(adj_matrix_norm) # TensorFlow-compatible sparse format



#! Graph Neural Network Learning Model
#? Build GNN Model
input_size = node_features_all.shape[1]
output_size = 2
hidden_layer_size = 32

class GNNModel(Model):
   def __init__(self):
      super().__init__()
      self.gcn1 = GCNConv(hidden_layer_size, activation='relu', input_shape=(input_size,))
      self.dropout1 = tf.keras.layers.Dropout(0.3)
      self.gcn2 = GCNConv(hidden_layer_size, activation='relu')
      self.dropout2 = tf.keras.layers.Dropout(0.3)
      self.out = tf.keras.layers.Dense(output_size, activation='softmax')

   def call(self, inputs):
      nfm, ajm = inputs  # nfm = node feature matrix, ajm = adjacency matrix
      nfm = self.gcn1([nfm, ajm])
      nfm = self.dropout1(nfm)
      nfm = self.gcn2([nfm, ajm])
      nfm = self.dropout2(nfm)
      return self.out(nfm)

#? Instantiate GNN Model
model = GNNModel()

#? Compile GNN Model
model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
   loss='sparse_categorical_crossentropy', 
   metrics=['accuracy'])

#? Set Learning Variables
batch_size = 100
max_epochs = 200

# How many nodes are in each subset
num_train = len(train_inputs)
num_val = len(validation_inputs)
num_test = len(test_inputs)

# Index masks for each subset
train_idx = np.arange(0, num_train)
val_idx = np.arange(num_train, num_train + num_val)
test_idx = np.arange(num_train + num_val, num_train + num_val + num_test)

# This makes a 1D array: 1.0 for training node indices, 0.0 elsewhere
sample_weight = np.isin(np.arange(len(labels_all)), train_idx).astype(float)

#? Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5, 
                                                  restore_best_weights=True, 
                                                  verbose=1)
#! Train the model
history = model.fit(x=(
                       node_features_all, 
                       similarity_adjacency_matrix), 
                    y=labels_all,
                    sample_weight=sample_weight,
                    batch_size=similarity_adjacency_matrix.shape[0], 
                    epochs=max_epochs,
                    callbacks=[early_stopping],
                    verbose=2)

#! Evaluate on test set
#? Prediction Results
predictions = model.predict((node_features_all, similarity_adjacency_matrix))
predicted_classes = np.argmax(predictions, axis=1)

# Slice for test nodes only
true_test_labels = labels_all[test_idx]
predicted_test_classes = predicted_classes[test_idx]

# Calculate test accuracy manually
from sklearn.metrics import accuracy_score
test_accuracy, test_loss = accuracy_score(true_test_labels, predicted_test_classes)

#? Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix and transpose it
cm = confusion_matrix(true_test_labels, predicted_test_classes)
cm = cm.T  # Make actual labels the column headers

# Labels
class_names = ['No Stroke', 'Stroke']

# Plot confusion matrix with actual on top
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Actual Label', labelpad=10)
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix (Actual on Top)', pad=20)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.tight_layout()
plt.show()

# Print classification report
print("\nClassification Report (Test Set Only):")
print(classification_report(true_test_labels, predicted_test_classes, target_names=class_names))

# Print accuracy
print(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')
