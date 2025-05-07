seed = 42

import random
random.seed(seed)

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Optional but helps enforce determinism
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['PYTHONHASHSEED'] = str(seed)

import numpy as np
np.random.seed(seed)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(seed)

import matplotlib.pyplot as plt
import pandas as pd

def ohe_replace_cols_at_pos(df, cols_list):
   for og_col_name in cols_list:
      # Create One-Hot Encoding for the given column
      ohe_df = ohe.fit_transform(csv_df[[og_col_name]])
      
      # Store index of original column positition
      index = df.columns.get_loc(og_col_name)
      
      # Update the old column with the new columns
      df = df.drop(columns=[og_col_name])
      for col in reversed(ohe_df.columns):
         df.insert(index, col, ohe_df[col])
         
   return df

#! Library Settings
#? Pandas, settings
pd.set_option('display.max_columns', None)

#? Turn randomness off by uncommenting these
np.random.seed(42)
tf.random.set_seed(42)



#! Preprocessing
#? Read the file's matrix to a var
# raw_csv_data = np.loadtxt('./final/healthcare-dataset-stroke-data-iterable.csv', delimiter=',')
raw_csv_data_location = './final/healthcare-dataset-stroke-data.csv'
csv_df = pd.read_csv(raw_csv_data_location, na_values='N/A')

#? Adjust data in table to be usable for our purposes
# One-Hot Encoding for categorical nominal data
ohe = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform='pandas')
csv_df = ohe_replace_cols_at_pos(csv_df, ['gender','ever_married','work_type','Residence_type','smoking_status'])

# Replace N/A for bmi column/feature and replace with median (as bmi is right skewed)
csv_df['bmi'] = pd.to_numeric(csv_df['bmi'], errors='coerce')
csv_df['bmi'] = csv_df['bmi'].fillna(csv_df['bmi'].median())

csv_df = csv_df.iloc[1:]

# Convert the formated pandas data frame to numpy
all_data = csv_df.to_numpy()

# # Remove the first row (it only contains the headers)
# all_data = np.delete(all_data, 0, axis=0)

# Remove unhelpful columns
unscaled_inputs_all = all_data[:,1:-1] # remove first and last columns
targets_all = all_data[:,-1] # saves targets to last column

#? Balancing the datasheet
num_one_targets = int(np.sum(targets_all))
zero_target_counter = 0

indicies_to_remove = []

# Reduce the number of majority class samples (0s) by trimming the excess, so they match the number of minority class samples (1s)
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_target_counter += 1
        if zero_target_counter > num_one_targets:
            indicies_to_remove.append(i)

# Remove excess 0s using the indices we collected
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indicies_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indicies_to_remove, axis=0)

#? Standardize Inputs [Sklearn section]
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
# scaled_inputs = unscaled_inputs_equal_priors

#? Shuffle the Data (just in case)
   # Shuffle indicies in the data
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

#? Print statistics
print(np.sum(train_targets), len(train_targets), np.sum(train_targets) / len(train_targets))
print(np.sum(validation_targets), len(validation_targets), np.sum(validation_targets) / len(validation_targets))
print(np.sum(test_targets), len(test_targets), np.sum(test_targets) / len(test_targets))



#! Deep Learning Model
#? Set the input and output sizes
input_size = 21
output_size = 2
hidden_layer_size = 32 # Same size for both hidden layers

model = tf.keras.Sequential([
   tf.keras.Input(shape=(input_size,)),
   tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
   loss='sparse_categorical_crossentropy', 
   metrics=['accuracy'])

batch_size = 100
max_epochs = 200

#? Early stopping to prevent overfitting, with patience 3
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

#? Confusion Matrix
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
print(classification_report(test_targets, predicted_classes, target_names=class_names))


print(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')