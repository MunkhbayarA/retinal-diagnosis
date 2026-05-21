import streamlit as st
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tempfile
import os
from recommendation import cnv, dme, drusen, normal

@st.cache_resource
def load_my_model():
    custom_objects = {'Addons>F1Score': tfa.metrics.F1Score}
    model_path = "Trained_Model.h5"
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def model_prediction(test_image_path):
    model = load_my_model()
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return np.argmax(predictions)

st.set_page_config(page_title="OCT Scan Analysis Platform", layout="wide")

st.sidebar.image("logo.png", width=250)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

if app_mode == "Home":
    background_image_path = "background.png"
    
    if os.path.exists(background_image_path):
        import base64
        with open(background_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .text-container {{
                background-color: rgba(0, 0, 0, 0.65); 
                padding: 30px; 
                border-radius: 15px;
                max-width: 800px;
                margin-top: 50px;
            }}
            .text-container h2 {{
                color: #FFFFFF !important;
            }}
            .text-container p {{
                color: #E5E7EB !important;
                font-size: 18px;
            }}
            </style>
            """, unsafe_allow_html=True)

    st.markdown('<div class="text-container"><h2>OCT Scan Analysis Platform</h2><p>Machine Learning 2025-2026 Project.</p></div>', unsafe_allow_html=True)

elif app_mode == "About":
    st.header("About the Project")
    st.markdown('''
### 👁️ Project Overview
This project is **OCT Scan Analysis**. It is built to help doctors scan and read eye medical images instantly. By uploading an eye scan to this app, the software automatically looks for damage and identifies specific eye conditions in seconds.

### 🧠 How It Works
* **The Brain:** The app uses a pre-trained artificial intelligence model (Mobile
