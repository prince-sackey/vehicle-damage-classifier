"""
Vehicle Damage Classifier — Streamlit deployment app.

Loads the model saved by the notebook (`vehicle_damage_mobilenetv2.keras` and
`class_names.json`) and serves a simple image-upload interface. Runs entirely
on CPU — no GPU queue, no ZeroGPU quota to wait on.

To deploy on Streamlit Community Cloud:
1. Push this file, requirements.txt, vehicle_damage_mobilenetv2.keras, and
   class_names.json to a GitHub repo (public or connected to your account).
2. Go to share.streamlit.io, sign in with GitHub, and create a new app
   pointing at this repo and this file (streamlit_app.py).
3. It builds and serves automatically.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import mobilenet_v2

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("vehicle_damage_mobilenetv2.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model_and_labels()

st.title("Vehicle Damage Classifier")
st.write(
    "Upload a photo of vehicle damage to classify it as a crack, dent, glass shatter, "
    "lamp broken, scratch, or tire flat. Built with a MobileNetV2 transfer-learning model "
    "reaching 84% test accuracy — see the training notebook for the full evaluation and "
    "confusion matrix."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_array = np.array(image)
    img_tensor = tf.image.resize(img_array, IMG_SIZE)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = mobilenet_v2.preprocess_input(img_tensor)

    probs = model.predict(img_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: {class_names[predicted_idx]}")
    st.write("Confidence by class:")

    results = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    for name, prob in results:
        st.write(f"{name}: {prob:.1%}")
        st.progress(float(prob))