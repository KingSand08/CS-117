from ml_settings import seed, random, os, np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib

cached = {}

#! Library Settings
#? Pandas, settings
pd.set_option('display.max_columns', None)

raw_csv_data_location = './final/healthcare-dataset-stroke-data.csv'

def get_headers():
    csv_df = pd.read_csv(raw_csv_data_location, na_values='N/A')

    # One-hot encode categorical columns
    processed_df = ohe_replace_cols_at_pos(csv_df.copy(), ['gender','ever_married','work_type','Residence_type','smoking_status'])

    # Ensure BMI is numeric
    processed_df['bmi'] = pd.to_numeric(processed_df['bmi'], errors='coerce')
    processed_df['bmi'] = processed_df['bmi'].fillna(processed_df['bmi'].median())

    return list(processed_df.columns)

# Prepare One-Hot Encoding
ohe = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform='pandas')

def ohe_replace_cols_at_pos(df, cols_list):
   for og_col_name in cols_list:
      # Create One-Hot Encoding for the given column
      ohe_df = ohe.fit_transform(df[[og_col_name]])
      
      # Store index of original column positition
      index = df.columns.get_loc(og_col_name)
      
      # Update the old column with the new columns
      df = df.drop(columns=[og_col_name])
      for col in reversed(ohe_df.columns):
         df.insert(index, col, ohe_df[col])
         
   return df


def format_data(cache=True):
    #! Preprocessing
    #? Read the file's matrix to a var
    # raw_csv_data = np.loadtxt('./final/healthcare-dataset-stroke-data-iterable.csv', delimiter=',')
    csv_df = pd.read_csv(raw_csv_data_location, na_values='N/A')

    #? Adjust data in table to be usable for our purposes
    # One-Hot Encoding for categorical nominal data
    csv_df = ohe_replace_cols_at_pos(csv_df, ['gender','ever_married','work_type','Residence_type','smoking_status'])

    # Replace N/A for bmi column/feature and replace with median (as bmi is right skewed)
    csv_df['bmi'] = pd.to_numeric(csv_df['bmi'], errors='coerce')
    csv_df['bmi'] = csv_df['bmi'].fillna(csv_df['bmi'].median())

    csv_df = csv_df.iloc[1:]

    # Convert the formated pandas data frame to numpy
    all_data = csv_df.to_numpy()
    #? SAVE TO FILE
    csv_df.to_csv('NEW_DATA.csv', index=True)


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
    scaler = preprocessing.StandardScaler()
    scaled_inputs = scaler.fit_transform(unscaled_inputs_equal_priors)
    joblib.dump(scaler, "models/scaler.pkl")

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
    
    # SMOKED HERE --> (SMARTER THAN RANDOM OVERSAMPLING)

    #? Print statistics
    # print(np.sum(train_targets), len(train_targets), np.sum(train_targets) / len(train_targets))
    # print(np.sum(validation_targets), len(validation_targets), np.sum(validation_targets) / len(validation_targets))
    # print(np.sum(test_targets), len(test_targets), np.sum(test_targets) / len(test_targets))
    
    split = (train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets)
    
    if cache:
        cached['split'] = split
        
    return split
