
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leaflens.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

st.set_page_config(
    page_title="LeafLens AI",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 LeafLens AI")
st.subheader("AI-Powered Plant Disease Detection")
st.write(
    "Upload a clear picture of a plant leaf to see "
    "which disease category the model predicts."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def get_guidance(class_name):
    label = class_name.replace("___", " - ").replace("_", " ")

    if "healthy" in class_name.lower():
        return (
            "The model classifies this leaf as healthy. "
            "Continue regular watering, monitoring, and plant care."
        )

    return (
        f"The model predicts: {label}. "
        "Inspect the plant and nearby leaves for similar symptoms. "
        "Avoid spreading possible infection by cleaning tools and "
        "handling affected plants carefully. Confirm the diagnosis "
        "with a local agricultural expert before applying treatment."
    )

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "Trained model files are missing. "
        "Run `python train.py` first, then reopen this app."
    )
    st.stop()

try:
    model = load_model()

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded leaf image",
            width="stretch",
        )

        if st.button("🔍 Detect Plant Disease", type="primary"):
            with st.spinner("Analyzing leaf image..."):
                resized_image = image.resize((160, 160))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32,
                )

                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index])

            st.success("Analysis completed!")

            st.markdown("### 🌱 Prediction")
            st.write(
                predicted_class.replace("___", " - ").replace("_", " ")
            )

            st.metric(
                "Model score",
                f"{confidence * 100:.2f}%",
            )

            if confidence < 0.60:
                st.warning(
                    "The model is uncertain about this prediction. "
                    "Try a clearer image and seek expert confirmation."
                )

            st.markdown("### 🩺 Crop-Care Guidance")
            st.info(get_guidance(predicted_class))

            st.caption(
                "This is an AI prediction, not a confirmed diagnosis. "
                "The model score is not a guarantee of accuracy."
            )

            # Show the top three predicted categories
            st.markdown("### 📊 Top Predictions")

            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:
                label = (
                    class_names[index]
                    .replace("___", " - ")
                    .replace("_", " ")
                )

                st.write(
                    f"{label}: {predictions[index] * 100:.2f}%"
                )
                st.progress(float(predictions[index]))

    except Exception as error:
        st.error(f"Could not analyze this image: {error}")

st.divider()
st.caption(
    "LeafLens AI | Educational deep learning mini project"
)
