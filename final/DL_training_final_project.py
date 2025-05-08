from ml_settings import np, tf, plt
from format_data import format_data as data

def dl_builder():
   #! Deep Learning Model
   #? Set the input and output sizes
   input_size = 21
   output_size = 2
   hidden_layer_size = 32 # Same size for both hidden layers

   dl_model = tf.keras.Sequential([
      tf.keras.Input(shape=(input_size,)),
      tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(output_size, activation='softmax')
   ])
   
   return dl_model

# Extract the data from the formatting data function
train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

# Obtain DL model
model = dl_builder()

#? Compile DL Model
model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
   loss='sparse_categorical_crossentropy', 
   metrics=['accuracy'])

#? Set Learning Variables
batch_size = 100
max_epochs = 200

#? Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5, 
                                                  restore_best_weights=True, 
                                                  verbose=1)



#! Train the model
history = model.fit(train_inputs,
                    train_targets,
                    batch_size=batch_size,
                    epochs=max_epochs,
                    callbacks=[early_stopping],
                    validation_data=(validation_inputs, validation_targets), 
                    verbose=2)



#! Evaluate on test set
#? Prediction Results
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=0)
predictions = model.predict(test_inputs)
predicted_classes = np.argmax(predictions, axis=1)

# #? Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Predict
predicted_probs = model.predict(test_inputs)
predicted_classes = np.argmax(predicted_probs, axis=1)

# Compute confusion matrix and transpose it
cm = confusion_matrix(test_targets, predicted_classes)
cm = cm.T  # Transpose so actuals are columns

# Labels
class_names = ['No Stroke', 'Stroke']

from sklearn.metrics import classification_report

# Print classification report
print("\nClassification Report (Test Set Only):")
print(classification_report(test_targets, predicted_classes, target_names=class_names))

print(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')

# Plot confusion matrix for DL model
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Deep Learning Confusion Matrix")
# plt.tight_layout()
# plt.show()
