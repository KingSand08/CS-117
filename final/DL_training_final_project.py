from sklearn.metrics import f1_score
from ml_settings import np, tf, plt
from format_data import format_data as data

# Returns a deep learning model

def dl_builder(hidden_size1, hidden_size2, dropout_rate1, dropout_rate2):
   #! Deep Learning Model
   #? Set the input and output sizes
   input_size = 21
   output_size = 2

   dl_model = tf.keras.Sequential([
      tf.keras.Input(shape=(input_size,)),
      tf.keras.layers.Dense(hidden_size1, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate1),
      tf.keras.layers.Dense(hidden_size2, activation='relu'), 
      tf.keras.layers.Dropout(dropout_rate2),
      tf.keras.layers.Dense(output_size, activation='softmax')
   ])
   
   return dl_model


def dl_builder_1(hidden_size1, hidden_size2, hidden_size3, dropout_rate1, dropout_rate2, dropout_rate3):
   #! Deep Learning Model
   #? Set the input and output sizes
   input_size = 21
   output_size = 2

   dl_model = tf.keras.Sequential([
      tf.keras.Input(shape=(input_size,)),
      tf.keras.layers.Dense(hidden_size1, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate1),
      tf.keras.layers.Dense(hidden_size2, activation='relu'), 
      tf.keras.layers.Dropout(dropout_rate2),
      tf.keras.layers.Dense(hidden_size3, activation='relu'), 
      tf.keras.layers.Dropout(dropout_rate3),
      tf.keras.layers.Dense(output_size, activation='softmax')
   ])
   
   return dl_model


# Run deep learning model given model from variable from dl_builder function
def run_dl(model, learning_rate_size, batch_size_n, patience_size):
   # Extract the data from the formatting data function
   train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

   #? Compile DL Model
   model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_size),
      loss='sparse_categorical_crossentropy', 
      metrics=['accuracy'])

   #? Set Learning Variables
   batch_size = batch_size_n
   max_epochs = 200

   #? Early stopping to prevent overfitting
   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=patience_size, 
                                                   restore_best_weights=True, 
                                                   verbose=0)



   #! Train the model
   history = model.fit(train_inputs,
                     train_targets,
                     batch_size=batch_size,
                     epochs=max_epochs,
                     callbacks=[early_stopping],
                     validation_data=(validation_inputs, validation_targets), 
                     verbose=0)



   #! Evaluate on test set
   #? Validation Accuracy
   best_val_acc = max(history.history['val_accuracy'])
   print(f"Validation accuracy: {best_val_acc:.4f}")

   #? Test Accuracy
   test_loss, test_acc = model.evaluate(test_inputs, test_targets, verbose=0)
   print(f"Test accuracy:       {test_acc:.4f}")

   #? F1 Stoke Score
   preds = model.predict(test_inputs, verbose=0)
   pred_classes = np.argmax(preds, axis=1)
   f1_stroke = f1_score(test_targets, pred_classes, pos_label=1)
   print(f"F1-score (Stroke=1): {f1_stroke:.4f}")
   

   return test_acc, test_loss, f1_stroke
