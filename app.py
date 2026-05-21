import streamlit as st
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tempfile
import os
from recommendation import cnv, dme, drusen, normal

# 1. Configure the model loading with cache and custom object registration
@st.cache_resource
def load_my_model():
    custom_objects = {'Addons>F1Score': tfa.metrics.F1Score}
    # Ensure this path is exactly where your .h5 file is located
    model_path = r"C:\Users\chess\Documents\Eye_Disease_Prediction\Dataset - train+val+test\Trained_Model.h5"
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# 2. Optimized Prediction Function
def model_prediction(test_image_path):
    model = load_my_model()
    # Load and resize
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0) # Shape: (1, 224, 224, 3)
    
    # Run prediction
    predictions = model.predict(x)
    return np.argmax(predictions)

# --- UI Setup ---
st.set_page_config(page_title="Eye Disease Prediction", layout="wide")

# ADD THIS LINE TO DISPLAY THE LOGO AT THE TOP OF THE SIDEBAR
st.sidebar.image(r"C:\Users\chess\Documents\Eye_Disease_Prediction\Dataset - train+val+test\logo.png", use_container_width=True)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# --- Pages ---
if app_mode == "Home":
    # --- BACKGROUND PHOTO ENGINE ---
    background_image_path = r"C:\Users\chess\Documents\Eye_Disease_Prediction\Dataset - train+val+test\background.png"
    
    if os.path.exists(background_image_path):
        import base64
        with open(background_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        
        # Inject custom CSS to handle the background image and text containers
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* This box creates a dark transparent shield so text is perfectly readable */
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
            """, unsafe_allow_html=True) # Fixed argument name here
    # ---------------------------------

    # HTML wrapper box using our new styles
    st.markdown('''
        <div class="text-container">
            <h2>OCT Scan Analysis Platform</h2>
            <p>Machine Learning 2025-2026 Project.</p>
        </div>
    ''', unsafe_allow_html=True) # Fixed argument name here

elif app_mode == "About":
    st.header("About the Project")
    
    st.markdown('''
### 👁️ Project Overview
This project is **OCT Scan Analysis**. It is built to help doctors scan and read eye medical images instantly. By uploading an eye scan to this app, the software automatically looks for damage and identifies specific eye conditions in seconds.



### 🧠 How It Works
* **The Brain:** The app uses a pre-trained artificial intelligence model (MobileNetV3) that has learned to recognize what healthy eyes look like versus diseased eyes.
* **The Process:** When you upload an image, the system processes it, extracts patterns from the retinal layers, and gives you the final matching prediction.
* **The Interface:** Built entirely with Python and Streamlit to give doctors a fast, simple web screen to run diagnostics locally.



### 📊 The 4 Conditions It Can Predict
The system is trained to identify four specific categories:

1. **CNV:** Abnormal blood vessels growing under the retina.
2. **DME:** Fluid buildup causing swelling in the eye.
3. **DRUSEN:** Small yellow spots or deposits forming under the retina.
4. **NORMAL:** Clear, healthy eyes with no visible diseases.


**Developer:** ANKHBAYAR MUNKHBAYAR  
**University:** Shanghai University of Engineering Science
''')

elif app_mode == "Disease Identification":
    st.header("Disease Identification")
    test_image = st.file_uploader("Upload your Image:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        # Create a stable temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(test_image.name)[1]) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
        
        if st.button("Predict"):
            with st.spinner("Analyzing scan..."):
                try:
                    result_index = model_prediction(temp_file_path)
                    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    prediction = class_names[result_index]
                    
                    st.success(f"Model prediction: **{prediction}**")
                    
                    # Logic for recommendations
                    recs = [cnv, dme, drusen, normal]
                    with st.expander("Learn More"):
                        st.image(test_image)
                        st.markdown(recs[result_index])
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)