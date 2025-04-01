import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model("disease_lstm_model.h5")

# Load training dataset for label encoding and scaling
train_df = pd.read_csv("Training.csv") 
if "Unnamed: 133" in train_df.columns:
    train_df = train_df.drop(columns=["Unnamed: 133"])

# Encode the target variable
label_encoder = LabelEncoder()
train_df['prognosis'] = label_encoder.fit_transform(train_df['prognosis'])

# Normalize the features
scaler = MinMaxScaler()
scaler.fit(train_df.drop(columns=['prognosis']))

# Streamlit UI
st.title("🩺 Disease Prediction using LSTM")
st.write("Select symptoms and get a predicted disease.")

# Get list of symptoms
symptom_columns = train_df.drop(columns=['prognosis']).columns.tolist()

# User input for symptoms
user_symptoms = {symptom: st.checkbox(symptom) for symptom in symptom_columns}

if st.button("Predict Disease"):
    # Convert input to model format
    input_data = np.array([int(user_symptoms[symptom]) for symptom in symptom_columns]).reshape(1, -1)
    
    # Normalize input
    input_data = scaler.transform(input_data)
    
    # Reshape for LSTM
    input_data = input_data.reshape(1, 1, -1)

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"🩺 Predicted Disease: **{predicted_disease}**")
