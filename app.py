import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

# Download the model from Hugging Face if not exists
MODEL_URL = "https://huggingface.co/YOUR_USERNAME/skin-cancer-model/resolve/main/model.h5"
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
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Streamlit UI
st.title("Skin Cancer Detection")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        prediction = predict_image(image)
        st.write(f"Prediction: {prediction}")
