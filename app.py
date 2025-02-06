import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

# Download the model from Hugging Face if not exists
MODEL_URL = "https://huggingface.co/mdshihabshorkar/skin-cancer-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading Model..."):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Prediction Function
def predict_image(image):
    # Resize the image to match the model's expected input shape
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize image
    
    # Check image shape for debugging
    st.write(f"Image Shape: {image.shape}")  # Debugging line
    
    # Convert grayscale to RGB if necessary
    if image.ndim == 2:  
        image = np.stack([image] * 3, axis=-1)
    
    # Add batch dimension (this should be 1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    
    st.write(f"Image Shape after expansion: {image.shape}")  # Debugging line
    
    # Model prediction
    prediction = model.predict(image)
    
    # Process prediction (e.g., for classification output)
    predicted_class = np.argmax(prediction, axis=-1)  # Get class index
    confidence = np.max(prediction)  # Confidence level
    
    return predicted_class, confidence

# Streamlit UI
st.title("Skin Cancer Detection")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        predicted_class, confidence = predict_image(image)
        st.write(f"Prediction: Class {predicted_class[0]}, Confidence: {confidence:.2f}")
