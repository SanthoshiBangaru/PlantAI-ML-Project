import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="PlantAI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_ml_model():
    return load_model("model/plant_disease_model.h5")

model = load_ml_model()

# ---------------------------------------------------
# LOAD LABELS
# ---------------------------------------------------

with open("model/labels.json") as f:
    labels = json.load(f)

# ---------------------------------------------------
# LOAD TREATMENT DATA
# ---------------------------------------------------

with open("model/treatment_data.json") as f:
    treatment_db = json.load(f)

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------

IMG_SIZE = 224

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_disease(image):

    processed = preprocess_image(image)

    predictions = model.predict(processed, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[predicted_index])

    class_name = labels.get(
        str(predicted_index),
        "Unknown Disease"
    )

    details = treatment_db.get(class_name, {
        "description": "No information available.",
        "treatment": ["No treatment available."],
        "sustainable_tips": ["No sustainable care tips available."]
    })

    return class_name, confidence, details

# ---------------------------------------------------
# PROFESSIONAL MINIMAL UI
# ---------------------------------------------------

st.markdown("""
<style>

/* App Background */

.stApp {
    background: linear-gradient(
        135deg,
        #f6f9f7 0%,
        #edf7ef 100%
    );
}

/* Main Layout */

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit Branding */

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 800;
    color: #14532d;
    margin-bottom: 0.4rem;
    letter-spacing: -2px;
}

/* Subtitle */

.sub-text {
    text-align: center;
    color: #4b5563;
    font-size: 1.05rem;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* Upload Box */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #cbd5e1;
    border-radius: 22px;
    padding: 1.4rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

/* Uploaded Image */

img {
    border-radius: 22px;
}

/* Analyze Button */

.stButton > button {
    width: 100%;
    background: #166534;
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.95rem;
    font-size: 1rem;
    font-weight: 600;
    transition: 0.2s ease;
    margin-top: 0.8rem;
}

.stButton > button:hover {
    background: #14532d;
    transform: translateY(-2px);
}

/* Result Card */

.result-card {
    background: white;
    padding: 2.2rem;
    border-radius: 28px;
    margin-top: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

/* Disease Title */

.disease-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}

/* Confidence */

.confidence {
    font-size: 1.15rem;
    font-weight: 700;
    color: #166534;
    margin-bottom: 1.5rem;
}

/* Section Title */

.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #14532d;
    margin-top: 1.8rem;
    margin-bottom: 0.7rem;
}

/* Text */

p, li, div {
    color: #1f2937;
    line-height: 1.7;
}

/* Footer */

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 3rem;
    font-size: 0.9rem;
}

/* Success Box */

[data-testid="stAlert"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        🌿 PlantAI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">
        AI-powered plant disease detection system using Deep Learning,
        TensorFlow and Computer Vision.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# IMAGE DISPLAY + PREDICTION
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if st.button("Analyze Plant"):

        with st.spinner("Analyzing plant health..."):

            disease, confidence, details = predict_disease(image)

        st.success("Analysis Completed Successfully")

        # Main Result Card
        with st.container():

            st.markdown(
                """
                <div style="
                background:white;
                padding:32px;
                border-radius:24px;
                box-shadow:0 8px 24px rgba(0,0,0,0.06);
                margin-top:20px;
                ">
                """,
                unsafe_allow_html=True
            )

        st.markdown(
        f"""
        <h1 style="
            color:#111827;
            font-size:38px;
            margin-bottom:10px;
        ">
            🌱 {disease.replace('___', ' - ')}
        </h1>
        """,
        unsafe_allow_html=True
    )

        st.markdown(
            f"""
            <div style="
                color:#166534;
                font-size:22px;
                font-weight:700;
                margin-bottom:10px;
            ">
                Confidence Score: {round(confidence * 100, 2)}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # Disease Description
        st.markdown("### Disease Description")

        st.info(details["description"])

        # Treatment Suggestions
        st.markdown("### Treatment Suggestions")

        for item in details["treatment"]:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    padding:14px 18px;
                    border-radius:14px;
                    margin-bottom:12px;
                    border-left:5px solid #166534;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);
                ">
                    ✅ {item}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Sustainable Tips
        st.markdown("### Sustainable Care Tips")

        for item in details["sustainable_tips"]:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    padding:14px 18px;
                    border-radius:14px;
                    margin-bottom:12px;
                    border-left:5px solid #15803d;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);
                ">
                    🌿 {item}
                </div>
                """,
                unsafe_allow_html=True
            )