import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

#? Read the file's matrix to a var
raw_csv_data = np.loadtxt('Business_case_dataset.csv', delimiter=',')

#?w Preprocessing
unscaled_inputs_all = raw_csv_data[:,1:-1] # remove first and last columns
targets_all = raw_csv_data[:,-1] # saves targets to last column


#? Balancing the datasheet
num_one_targets = int(np.sum(targets_all))
zero_target_counter = 0

indicies_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_target_counter += 1
        if zero_target_counter > num_one_targets:
            indicies_to_remove.append(i)
            
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indicies_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indicies_to_remove, axis=0)

#? Standardize Inputs [Sklearn section]
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
# scaled_inputs = unscaled_inputs_equal_priors

#? Shuffle the Data (data is originally arranged by date, so this is a good approach, but usually is a good approach to take anyways )
   # Shuffle indicies in the data (based on date)
shuffled_indicies = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indicies)

   # Use shuffled indicies to shuffle the inputs and targets (ensures target and date maintain their original records aka rows since we shuffled the former)
shuffled_inputs = scaled_inputs[shuffled_indicies]
shuffled_targets = targets_equal_priors[shuffled_indicies]


#? Splitting the data
# Split the dataset into train and temp sets
train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(
shuffled_inputs, shuffled_targets, test_size=0.2, random_state=42
)

# Split the temp set into validation (50% of temp) and test (50% of temp) sets
validation_inputs, test_inputs, validation_targets, test_targets = train_test_split(
temp_inputs, temp_targets, test_size=0.5, random_state=42
)

# Print statistics
print(np.sum(train_targets), len(train_targets), np.sum(train_targets) / len(train_targets))
print(np.sum(validation_targets), len(validation_targets), np.sum(validation_targets) / len(validation_targets))
print(np.sum(test_targets), len(test_targets), np.sum(test_targets) / len(test_targets))

#! Deep Learning Model
# Set the input and output sizes
input_size = 10
output_size = 2
hidden_layer_size = 100 # Same size for both hidden layers

model = tf.keras.Sequential([
   tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(input_size,)),
   tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'), 
   tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
   loss='sparse_categorical_crossentropy', 
   metrics=['accuracy'])

batch_size = 100
max_epochs = 100

#? Early stopping to prevent overfitting, with patience 3
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=3, 
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


#! Plot the data
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()