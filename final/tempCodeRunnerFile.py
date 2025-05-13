from format_data import ohe_replace_cols_at_pos, preprocessing, pd, get_headers
import joblib

raw_input_row = {
    'gender': 'Male',
    'age': 22,
    'hypertension': 0,
    'heart_disease': 0,
    'ever_married': 'No',
    'work_type': 'Never_worked',
    'Residence_type': 'Urban',
    'avg_glucose_level': 90,
    'bmi': 25.1,
    'smoking_status': 'never smoked',
}


raw_input_row1 = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228,
    'bmi': 69,
    'smoking_status': 'formerly smoked',
}

def preprocess_new_input(raw_dict):
    df = pd.DataFrame([raw_dict])
    
    # One-hot encode using the live encoder
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df = ohe_replace_cols_at_pos(df, cat_cols)
    
    # Clean up BMI
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Get expected column order from training
    expected_cols = get_headers()

    # Add missing columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_cols]

    # Drop non-feature columns
    df = df.drop(columns=['id', 'stroke'])

    # Scale
    scaler = joblib.load("models/scaler.pkl")
    return scaler.transform(df.to_numpy())


# input_data = (0,1,0,0,22,0,0,1,0,0,1,0,0,0,0,1,85,25.1,0,0,1,0)
scaled_new_data = preprocess_new_input(raw_input_row1)

loaded_dl_bagging_model = joblib.load("models/dl_bagging_model.pkl")

prediction = loaded_dl_bagging_model.predict(scaled_new_data)
proba = loaded_dl_bagging_model.predict_proba(scaled_new_data)

print("Final processed input shape:", scaled_new_data.shape)
print("First few values:", scaled_new_data[0][:10])
print(prediction)
print("Probabilities:", proba)

