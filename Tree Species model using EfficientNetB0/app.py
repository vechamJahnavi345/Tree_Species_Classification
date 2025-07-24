import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model
import gdown
import os

@st.cache_resource
def load_model_from_gdrive():
    url = 'https://drive.google.com/file/d/1s3gZPsaW0krUQd2oqkBM-cfrp8lv66HW/view?usp=drive_link'
    output_path = 'tree_species_model.h5'
    
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    
    return load_model(output_path)

model = load_model_from_gdrive()

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))  # Use your model's input size
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if needed

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Example class names — replace with yours
    class_names = ["Neem", "Banyan", "Peepal", "Mango", "Gulmohar"]
    st.success(f"🌿 Predicted Species: **{class_names[predicted_class]}**")

