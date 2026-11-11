
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leaflens.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
TEST_DIR = BASE_DIR / "dataset" / "test"

IMAGE_SIZE = (160, 160)
CONFIDENCE_WARNING = 60.0

st.set_page_config(
    page_title="LeafLens AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# 2. CUSTOM UI DESIGN
# ==================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --green: #246b45;
        --green-dark: #174b32;
        --green-light: #e9f5ed;
        --green-soft: #f4f8f2;
        --text: #203329;
        --muted: #718078;
        --border: #e1e9e1;
        --white: #ffffff;
    }

    .stApp {
        background: #f5f8f4;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 244, 0.92);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--green-dark) !important;
        letter-spacing: -0.5px;
    }

    p, label, li {
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #edf4ec;
        border-right: 1px solid #dce8dc;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #4e6555;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 22px 0;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 49px;
        height: 49px;
        border-radius: 15px;
        background: #246b45;
        color: white;
        font-size: 27px;
        flex-shrink: 0;
    }

    .brand-name {
        color: #174b32;
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: #718078;
        font-size: 12px;
        margin-top: 4px;
    }

    .hero {
        padding: 30px;
        border-radius: 23px;
        background: linear-gradient(120deg, #174b32 0%, #246b45 65%, #3b8054 100%);
        color: white;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    .hero-kicker {
        color: #c8e6cf;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(27px, 4vw, 39px);
        line-height: 1.2;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #e0eee3;
        font-size: 15px;
        max-width: 680px;
        line-height: 1.7;
        margin: 0;
    }

    .hero-leaf {
        font-size: 88px;
        text-align: center;
        opacity: 0.9;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        font-weight: 800;
        color: #174b32;
        margin: 25px 0 5px 0;
    }

    .section-subtitle {
        color: #718078;
        font-size: 13px;
        margin-bottom: 17px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e1e9e1;
        border-radius: 17px;
        padding: 19px;
        min-height: 115px;
        box-shadow: 0 3px 12px rgba(29, 72, 45, 0.035);
    }

    .metric-label {
        font-size: 12px;
        color: #718078;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        color: #174b32;
        font-size: 26px;
        font-weight: 800;
        line-height: 1.25;
    }

    .metric-note {
        font-size: 11px;
        color: #718078;
        margin-top: 7px;
    }

    .panel {
        background: white;
        border: 1px solid #e1e9e1;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .result-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #718078;
        margin-bottom: 7px;
    }

    .result-name {
        font-family: 'Manrope', sans-serif;
        color: #174b32;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.4;
        margin-bottom: 12px;
    }

    .confidence-pill {
        display: inline-block;
        background: #e6f4e9;
        color: #246b45;
        border-radius: 30px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 700;
    }

    .info-card {
        background: #f0f7f0;
        border-left: 4px solid #4b8c5d;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 12px 0;
    }

    .info-card p {
        margin: 0;
        line-height: 1.65;
        font-size: 14px;
    }

    .small-muted {
        font-size: 12px;
        color: #718078;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 11px;
        border: 1px solid #246b45;
        font-weight: 700;
        min-height: 43px;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"] {
        background: #246b45;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #174b32;
        border-color: #174b32;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #a9c8ae;
        border-radius: 15px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e1e9e1;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #849187;
        font-size: 12px;
        border-top: 1px solid #e1e9e1;
        padding-top: 20px;
        margin-top: 35px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px;
        }

        .hero-leaf {
            font-size: 45px;
        }

        .metric-value {
            font-size: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 3. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found at {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def pretty_label(label):
    return (
        label.replace("___", " - ")
        .replace("__", " - ")
        .replace("_", " ")
        .strip()
    )


def normalize_name(name):
    return "".join(
        character.lower()
        for character in name
        if character.isalnum()
    )


def prepare_image(image):
    """
    The saved model includes Rescaling(1/127.5, offset=-1).
    Therefore, supply RGB pixel values in the 0-255 range.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(image):
    image_array = prepare_image(image)
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        {
            "Disease": pretty_label(class_names[int(index)]),
            "Confidence (%)": round(
                float(probabilities[index]) * 100, 2
            ),
        }
        for index in top_indices
    ]

    return (
        predicted_index,
        pretty_label(class_names[predicted_index]),
        confidence,
        top_predictions,
    )


DISEASE_INFO = {
    "Apple___Apple_scab": (
        "Dark olive or brown spots on leaves.",
        "Remove infected fallen leaves and improve airflow.",
    ),
    "Apple___Cedar_apple_rust": (
        "Yellow-orange spots on apple leaves.",
        "Remove affected plant debris and seek local plant-health guidance.",
    ),
    "Corn_(maize)___Common_rust_": (
        "Reddish-brown raised pustules on leaves.",
        "Monitor the spread and consult local agricultural guidance.",
    ),
    "Potato___Early_blight": (
        "Brown spots that may form concentric rings.",
        "Remove affected leaves and avoid prolonged leaf wetness.",
    ),
    "Potato___Late_blight": (
        "Dark, water-soaked patches that may spread rapidly.",
        "Seek agricultural advice promptly because the disease can spread quickly.",
    ),
    "Potato___healthy": (
        "No disease symptoms identified by the model.",
        "Continue regular monitoring and appropriate plant care.",
    ),
    "Tomato___Early_blight": (
        "Brown leaf spots, sometimes with concentric rings.",
        "Remove affected leaves and avoid overhead watering.",
    ),
    "Tomato___Late_blight": (
        "Dark, rapidly spreading patches on leaves.",
        "Seek local agricultural guidance promptly.",
    ),
    "Tomato___healthy": (
        "No disease symptoms identified by the model.",
        "Continue regular monitoring and appropriate plant care.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Yellowing, upward curling leaves and stunted growth.",
        "Monitor whiteflies and seek local agricultural guidance.",
    ),
}


def get_disease_info(label):
    target = normalize_name(label)

    for original_label, information in DISEASE_INFO.items():
        if normalize_name(original_label) == target:
            return information

    return (
        "No specific information is available for this class.",
        "Use this prediction as a preliminary indication and consult an agricultural expert if symptoms persist.",
    )


# ==================================================
# 5. TEST DATA LABEL MAPPING
# ==================================================

PREFIX_MAPPING = {
    "AppleCedarRust": "Apple___Cedar_apple_rust",
    "AppleScab": "Apple___Apple_scab",
    "AppleHealthy": "Apple___healthy",
    "CornCommonRust": "Corn_(maize)___Common_rust_",
    "CornHealthy": "Corn_(maize)___healthy",
    "CornNorthernLeafBlight": "Corn_(maize)___Northern_Leaf_Blight",
    "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoLateBlight": "Potato___Late_blight",
    "PotatoHealthy": "Potato___healthy",
    "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoLateBlight": "Tomato___Late_blight",
    "TomatoHealthy": "Tomato___healthy",
    "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "TomatoBacterialSpot": "Tomato___Bacterial_spot",
    "TomatoSpiderMites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "TomatoTargetSpot": "Tomato___Target_Spot",
    "TomatoLeafMold": "Tomato___Leaf_Mold",
    "TomatoSeptoriaLeafSpot": "Tomato___Septoria_leaf_spot",
}


def infer_true_label(filename):
    stem = Path(filename).stem

    for prefix in sorted(PREFIX_MAPPING, key=len, reverse=True):
        if stem.startswith(prefix):
            expected = normalize_name(PREFIX_MAPPING[prefix])

            for index, name in enumerate(class_names):
                if normalize_name(name) == expected:
                    return index

            return None

    return None


# ==================================================
# 6. SESSION STATE
# ==================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


# ==================================================
# 7. SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🌿</div>
            <div>
                <div class="brand-name">LeafLens AI</div>
                <div class="brand-subtitle">Plant Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Leaf Analysis",
            "Disease Information",
            "Prediction History",
            "Model Evaluation",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**MODEL STATUS**")
    st.success("Model loaded successfully")

    st.markdown(
        f"""
        <div class="small-muted">
        Classes available: <b>{len(class_names)}</b><br>
        Input size: <b>160 × 160</b><br>
        Model: <b>MobileNetV2</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("LeafLens AI · Educational project")


# ==================================================
# 8. REUSABLE PAGE HEADER
# ==================================================

def page_header(title, subtitle, icon="🌿"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">LEAFLENS AI · PLANT HEALTH</div>
            <div style="display:flex;align-items:center;gap:20px;">
                <div style="flex:1;">
                    <h1>{icon} {title}</h1>
                    <p>{subtitle}</p>
                </div>
                <div class="hero-leaf">🌱</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# 9. LEAF ANALYSIS
# ==================================================

if page == "Leaf Analysis":
    page_header(
        "Plant Disease Detection",
        "Upload a leaf photo to identify a possible disease and explore helpful plant-care information.",
        "🌿",
    )

    history = st.session_state.prediction_history

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "AVAILABLE CLASSES",
            str(len(class_names)),
            "Disease and healthy categories",
        )

    with col2:
        metric_card(
            "SAVED PREDICTIONS",
            str(len(history)),
            "In this session",
        )

    with col3:
        metric_card(
            "ANALYSIS MODE",
            "AI-powered",
            "MobileNetV2 classifier",
        )

    section_heading(
        "Analyze a leaf",
        "For best results, use a clear photo with the leaf in focus.",
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
        key="leaf_upload",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <h3>🌱 Ready to examine your plant?</h3>
                <p>
                Upload a leaf image to see the predicted class,
                confidence score, top predictions, and general guidance.
                </p>
                <p class="small-muted">
                Tip: Avoid blurry images, strong shadows, and backgrounds
                that hide the leaf.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            image_col, result_col = st.columns(
                [1, 1.15],
                gap="large",
            )

            with image_col:
                st.markdown("#### Uploaded image")
                st.image(
                    image,
                    use_container_width=True,
                )
                st.caption(
                    f"File: {uploaded_file.name} · "
                    f"{image.width} × {image.height} px"
                )

            with result_col:
                with st.spinner("Analyzing your leaf..."):
                    (
                        predicted_index,
                        predicted_label,
                        confidence,
                        top_predictions,
                    ) = predict_image(image)

                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="result-label">Predicted class</div>
                        <div class="result-name">{predicted_label}</div>
                        <span class="confidence-pill">
                            Confidence · {confidence:.2f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if confidence < CONFIDENCE_WARNING:
                    st.warning(
                        "Confidence is below 60%. Treat this prediction "
                        "with caution and try a clearer image."
                    )
                else:
                    st.success(
                        "The model has a relatively high confidence "
                        "in this prediction. It is not a confirmed diagnosis."
                    )

                st.markdown("#### Top 3 predictions")
                top_df = pd.DataFrame(top_predictions)
                st.dataframe(
                    top_df,
                    hide_index=True,
                    use_container_width=True,
                )

                info_symptoms, info_advice = get_disease_info(
                    class_names[predicted_index]
                )

                st.markdown("#### 🌱 What to look for")
                st.write(info_symptoms)

                st.markdown("#### 💡 General guidance")
                st.write(info_advice)

                if st.button(
                    "＋ Save to prediction history",
                    type="primary",
                    use_container_width=True,
                ):
                    existing = any(
                        item["Image"] == uploaded_file.name
                        and item["Predicted Disease"] == predicted_label
                        for item in st.session_state.prediction_history
                    )

                    if existing:
                        st.info(
                            "This image and prediction are already saved "
                            "in this session."
                        )
                    else:
                        st.session_state.prediction_history.insert(
                            0,
                            {
                                "Image": uploaded_file.name,
                                "Predicted Disease": predicted_label,
                                "Confidence (%)": round(confidence, 2),
                            },
                        )
                        st.success("Prediction saved!")
                        st.rerun()

        except Exception as error:
            st.error(f"Unable to analyze the image: {error}")

    st.info(
        "LeafLens AI is an educational project. Its predictions are not "
        "a substitute for expert plant diagnosis."
    )


# ==================================================
# 10. DISEASE INFORMATION
# ==================================================

elif page == "Disease Information":
    page_header(
        "Disease Library",
        "Explore the model's plant categories, common symptoms, and general care guidance.",
        "📚",
    )

    section_heading(
        "Explore a class",
        "Choose one of the classes used by your trained model.",
    )

    selected_class = st.selectbox(
        "Plant disease or healthy class",
        class_names,
        format_func=pretty_label,
    )

    symptoms, advice = get_disease_info(selected_class)

    st.markdown(
        f"""
        <div class="panel">
            <div class="result-label">Selected category</div>
            <div class="result-name">{pretty_label(selected_class)}</div>
            <p class="small-muted">
            Information provided here is general educational guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col1, info_col2 = st.columns(2, gap="large")

    with info_col1:
        st.markdown(
            f"""
            <div class="panel">
                <h3>🔎 Common symptoms</h3>
                <p>{symptoms}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        st.markdown(
            f"""
            <div class="panel">
                <h3>🌱 General guidance</h3>
                <p>{advice}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("All model classes")

    search = st.text_input(
        "Search classes",
        placeholder="Type a plant or disease name...",
    )

    visible_classes = [
        pretty_label(name)
        for name in class_names
        if search.lower() in pretty_label(name).lower()
    ]

    st.caption(f"{len(visible_classes)} matching classes")

    st.dataframe(
        pd.DataFrame({"Plant / disease class": visible_classes}),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "If a plant appears severely affected, confirm the condition with "
        "an agricultural expert before deciding on treatment."
    )


# ==================================================
# 11. PREDICTION HISTORY
# ==================================================

elif page == "Prediction History":
    page_header(
        "Prediction History",
        "Review the predictions you saved during this Streamlit session.",
        "📋",
    )

    history = st.session_state.prediction_history

    col1, col2 = st.columns(2)

    with col1:
        metric_card(
            "TOTAL SAVED",
            str(len(history)),
            "Predictions in this session",
        )

    with col2:
        if history:
            average_confidence = np.mean(
                [item["Confidence (%)"] for item in history]
            )
            metric_card(
                "AVERAGE CONFIDENCE",
                f"{average_confidence:.2f}%",
                "Across saved predictions",
            )
        else:
            metric_card(
                "AVERAGE CONFIDENCE",
                "—",
                "No predictions saved yet",
            )

    section_heading(
        "Saved predictions",
        "Your history is temporary and resets when the session is lost.",
    )

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        download_col, clear_col = st.columns(2)

        with download_col:
            st.download_button(
                "↓ Download history as CSV",
                data=csv_data,
                file_name="leaflens_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with clear_col:
            if st.button(
                "Clear prediction history",
                use_container_width=True,
            ):
                st.session_state.prediction_history = []
                st.rerun()

    else:
        st.markdown(
            """
            <div class="panel">
                <h3>🌿 No saved predictions yet</h3>
                <p>
                Your analyzed images will appear here after you save
                a prediction from the Leaf Analysis page.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Go to Leaf Analysis",
            type="primary",
        ):
            st.session_state["navigation_hint"] = "Leaf Analysis"
            st.rerun()


# ==================================================
# 12. MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":
    page_header(
        "Model Performance",
        "Evaluate the saved classifier on test images with recognizable filename labels.",
        "📊",
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Before you evaluate</h3>
            <p>
            The test images are labeled using filename prefixes in the app.
            Make sure each filename correctly identifies the actual class.
            This is a limited test and may not represent all model classes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "▶ Run model evaluation",
        type="primary",
        use_container_width=True,
    ):
        if not TEST_DIR.exists():
            st.error(f"Test directory not found: {TEST_DIR}")
            st.stop()

        extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }

        test_files = sorted(
            file for file in TEST_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in extensions
        )

        if not test_files:
            st.error("No supported images found in dataset/test.")
            st.stop()

        y_true = []
        y_pred = []
        results = []
        skipped_files = []

        progress = st.progress(0)
        status = st.empty()

        for position, image_path in enumerate(test_files, start=1):
            status.write(
                f"Evaluating image {position} of {len(test_files)}: "
                f"{image_path.name}"
            )

            try:
                true_index = infer_true_label(image_path.name)

                if true_index is None:
                    skipped_files.append(
                        f"{image_path.name} — no matching expected label"
                    )
                    progress.progress(position / len(test_files))
                    continue

                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")

                probabilities = model.predict(
                    prepare_image(image),
                    verbose=0,
                )[0]

                predicted_index = int(np.argmax(probabilities))
                confidence = float(
                    probabilities[predicted_index]
                ) * 100

                y_true.append(true_index)
                y_pred.append(predicted_index)

                results.append(
                    {
                        "Image": image_path.name,
                        "Actual Class": pretty_label(
                            class_names[true_index]
                        ),
                        "Predicted Class": pretty_label(
                            class_names[predicted_index]
                        ),
                        "Confidence (%)": round(confidence, 2),
                        "Correct": true_index == predicted_index,
                    }
                )

            except Exception as error:
                skipped_files.append(
                    f"{image_path.name} — {error}"
                )

            progress.progress(position / len(test_files))

        progress.empty()
        status.empty()

        if not y_true:
            st.error(
                "No images could be evaluated. Check PREFIX_MAPPING "
                "and the test filenames."
            )
            if skipped_files:
                st.write("Skipped files:")
                st.code("\n".join(skipped_files))
            st.stop()

        accuracy = accuracy_score(y_true, y_pred)
        correct_count = sum(
            actual == predicted
            for actual, predicted in zip(y_true, y_pred)
        )

        # Include predicted-only classes so false positives are counted.
        shown_labels = sorted(set(y_true) | set(y_pred))

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=shown_labels,
            target_names=[
                pretty_label(class_names[index])
                for index in shown_labels
            ],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).transpose().round(4)
        results_df = pd.DataFrame(results)

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=shown_labels,
        )

        st.session_state.evaluation_result = {
            "evaluated": len(y_true),
            "accuracy": accuracy,
            "correct": correct_count,
            "incorrect": len(y_true) - correct_count,
            "actual_class_count": len(set(y_true)),
            "report_df": report_df,
            "results_df": results_df,
            "matrix": matrix,
            "labels": shown_labels,
            "skipped": skipped_files,
        }

    # Render the latest saved evaluation even after download-button clicks.
    evaluation = st.session_state.evaluation_result

    if evaluation is not None:
        st.success("Evaluation completed successfully.")

        st.caption(
            f"Classes represented in actual labels: "
            f"{evaluation['actual_class_count']} of {len(class_names)}"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            metric_card(
                "EVALUATED IMAGES",
                str(evaluation["evaluated"]),
                "Images with inferred labels",
            )

        with m2:
            metric_card(
                "ACCURACY",
                f"{evaluation['accuracy'] * 100:.2f}%",
                "On this test sample",
            )

        with m3:
            metric_card(
                "CORRECT",
                str(evaluation["correct"]),
                "Correct predictions",
            )

        with m4:
            metric_card(
                "INCORRECT",
                str(evaluation["incorrect"]),
                "Incorrect predictions",
            )

        section_heading(
            "Classification report",
            "Precision, recall and F1-score for classes present in actual or predicted labels.",
        )

        st.dataframe(
            evaluation["report_df"],
            use_container_width=True,
        )

        report_csv = evaluation["report_df"].to_csv().encode("utf-8")

        st.download_button(
            "↓ Download classification report",
            data=report_csv,
            file_name="leaflens_classification_report.csv",
            mime="text/csv",
            key="download_report",
        )

        section_heading(
            "Individual test results",
            "Compare the inferred actual class with the model's prediction.",
        )

        st.dataframe(
            evaluation["results_df"],
            use_container_width=True,
            hide_index=True,
        )

        results_csv = evaluation["results_df"].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "↓ Download individual results",
            data=results_csv,
            file_name="leaflens_test_results.csv",
            mime="text/csv",
            key="download_results",
        )

        section_heading(
            "Confusion matrix",
            "Rows show actual classes; columns show predicted classes.",
        )

        labels = evaluation["labels"]
        display_names = [
            pretty_label(class_names[index])
            for index in labels
        ]

        width = max(8, len(labels) * 1.0)
        height = max(6, len(labels) * 0.8)

        figure, axis = plt.subplots(
            figsize=(width, height)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=evaluation["matrix"],
            display_labels=display_names,
        )

        display.plot(
            ax=axis,
            xticks_rotation=90,
            cmap="Greens",
            colorbar=False,
            values_format="d",
        )

        axis.set_title("Actual vs Predicted Classes")
        figure.tight_layout()

        st.pyplot(figure)
        plt.close(figure)

        if evaluation["skipped"]:
            with st.expander(
                f"Skipped images ({len(evaluation['skipped'])})"
            ):
                st.code("\n".join(evaluation["skipped"]))

        st.caption(
            "This evaluation uses a limited test sample. Filename-derived "
            "labels must be verified, and results should not be treated "
            "as full-dataset performance."
        )

    else:
        st.info(
            "Select 'Run model evaluation' to calculate the metrics, "
            "view the report, and generate the confusion matrix."
        )


# ==================================================
# 13. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        <b>🌿 LeafLens AI</b> · Plant Disease Classification<br>
        Educational project · Predictions should be independently verified
    </div>
    """,
    unsafe_allow_html=True,
)
