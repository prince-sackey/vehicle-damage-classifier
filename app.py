"""
Vehicle Damage Classifier — deployment app.

Loads the model saved by the notebook (`vehicle_damage_mobilenetv2.keras` and
`class_names.json`) and serves a simple image-upload interface with Gradio.

To deploy on Hugging Face Spaces:
1. Create a new Space (SDK: Gradio).
2. Upload this file as app.py, plus vehicle_damage_mobilenetv2.keras,
   class_names.json, and requirements.txt.
3. The Space builds and serves automatically — no extra config needed.
"""

import json

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("vehicle_damage_mobilenetv2.keras")

with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)


def predict(image):
    if image is None:
        return {}

    img = tf.image.resize(image, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)
    img = mobilenet_v2.preprocess_input(img)

    probs = model.predict(img, verbose=0)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a photo of the vehicle damage"),
    outputs=gr.Label(num_top_classes=6, label="Predicted damage type"),
    title="Vehicle Damage Classifier",
    description=(
        "Upload a photo of vehicle damage to classify it as a crack, dent, glass shatter, "
        "lamp broken, scratch, or tire flat. Built with a MobileNetV2 transfer-learning model — "
        "see the training notebook for evaluation details and confusion matrix."
    ),
)

if __name__ == "__main__":
    demo.launch()
