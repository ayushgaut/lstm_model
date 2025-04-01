import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the trained model
try:
    model = tf.keras.models.load_model("disease_lstm_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load training dataset for label encoding and scaling
try:
    train_df = pd.read_csv("Training.csv")
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Ensure the dataset contains 'prognosis' column
if 'prognosis' not in train_df.columns:
    st.error("Dataset does not contain 'prognosis' column.")
    st.stop()

# Encode the target variable
label_encoder = LabelEncoder()
train_df['prognosis'] = label_encoder.fit_transform(train_df['prognosis'])

# Normalize the features
scaler = MinMaxScaler()
X_train = train_df.drop(columns=['prognosis'])
scaler.fit(X_train)

# Streamlit UI
st.title("🩺 Disease Prediction using LSTM")
st.write("Select symptoms and get a predicted disease.")

# Get list of symptoms
symptom_columns = X_train.columns.tolist()

# User input for symptoms
user_symptoms = {symptom: st.checkbox(symptom) for symptom in symptom_columns}

if st.button("Predict Disease"):
    try:
        # Convert input to model format
        input_data = np.array([int(user_symptoms[symptom]) for symptom in symptom_columns]).reshape(1, -1)

        # Normalize input
        input_data = scaler.transform(input_data)

        # Ensure correct shape for LSTM model
        input_data = input_data.reshape(1, 1, input_data.shape[1])

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Check if predicted class is valid
        if predicted_class >= len(label_encoder.classes_):
            st.error("Invalid prediction: predicted class out of range.")
        else:
            predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
            st.success(f"🩺 Predicted Disease: **{predicted_disease}**")
    
    except Exception as e:
        st.error(f"Prediction Error: {e}")
