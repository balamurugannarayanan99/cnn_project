import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

st.set_page_config(page_title="AI vs Real Image Detection", layout="centered")

st.title("🤖 AI vs Real Image Detection")
st.write("Upload an image to check whether it is **AI-generated** or **Real**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    # prediction output 
    if prediction > 0.5: 
        st.success("🧠 Result: **AI Generated Image**") 
    else: 
        st.success("📷 Result: **Real Image**")
    st.write(f"Confidence Score: {prediction:.2f}")