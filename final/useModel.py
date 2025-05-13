from format_data import ohe_replace_cols_at_pos, preprocessing, pd, get_headers
import tensorflow as tf
import joblib

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

def predict_with_model(model, raw_dict, is_keras=False):
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
    scaled_new_data = scaler.transform(df.to_numpy())

    # print("Final processed input shape:", scaled_new_data.shape)
    # print("First few values:", scaled_new_data[0][:10])
    
    if is_keras:
        prediction = model.predict(scaled_new_data, verbose=0)
        predicted_class = tf.argmax(prediction, axis=1).numpy()
        print("Prediction:", predicted_class)
        print("Probabilities:", prediction)
    else:
        prediction = model.predict(scaled_new_data)
        try:
            proba = model.predict_proba(scaled_new_data)
            print("Prediction:", prediction)
            print("Probabilities:", proba)
        except AttributeError:
            print("Prediction:", prediction)
            print("Probabilities: N/A (voting='hard')")



def prettyRunModelTest(user_data):
    print('=' * 20 + ' PREDICTING MODELS' + '=' * 20)
    print('-' * 10 + ' DEEP LEARNING MODEL ' + '-' * 10)
    loaded_dl_model = tf.keras.models.load_model("models/dl_model.keras")
    predict_with_model(loaded_dl_model, user_data, True)

    print('-' * 10 + ' K-NN MODEL ' + '-' * 10)
    loaded_knn_model = joblib.load("models/knn_model.pkl")
    predict_with_model(loaded_knn_model, user_data)

    print('-' * 10 + ' VOTING MODEL ' + '-' * 10)
    loaded_voting_model = joblib.load("models/voting_model.pkl")
    predict_with_model(loaded_voting_model, user_data)

    print('-' * 10 + ' ENSEMBLE BAGGING (DL) MODEL ' + '-' * 10)
    loaded_dl_bagging_model = joblib.load("models/dl_bagging_model.pkl")
    predict_with_model(loaded_dl_bagging_model, user_data)

    print('-' * 10 + ' ENSEMBLE BAGGING (KNN) MODEL ' + '-' * 10)
    loaded_knn_bagging_model = joblib.load("models/knn_bagging_model.pkl")
    predict_with_model(loaded_knn_bagging_model, user_data)
    
prettyRunModelTest(raw_input_row1)