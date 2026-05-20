import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_ml_model():

    return load_model(
        "model/plant_disease_model.h5",
        compile=False,
        safe_mode=False
    )

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

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    confidence = float(
        predictions[predicted_index]
    )

    class_name = labels.get(
        str(predicted_index),
        "Unknown Disease"
    )

    details = treatment_db.get(
        class_name,
        {
            "description": "No information available.",
            "treatment": ["No treatment available."],
            "sustainable_tips": ["No sustainable care tips available."]
        }
    )

    return class_name, confidence, details

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f6f9f7 0%,
            #edf7ef 100%
        );
    }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .main-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 0.3rem;
    }

    .sub-text {
        text-align: center;
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        border: 2px dashed #cbd5e1;
    }

    .stButton > button {
        width: 100%;
        background: #166534;
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.9rem;
        font-size: 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #14532d;
    }

    img {
        border-radius: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

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
        AI-powered plant disease detection using Deep Learning and Computer Vision
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
# PREDICTION SECTION
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        width=700
    )

    if st.button("Analyze Plant"):

        with st.spinner(
            "Analyzing plant health..."
        ):

            disease, confidence, details = predict_disease(image)

        st.success(
            "Analysis Completed Successfully"
        )

        # ---------------------------------------------------
        # RESULT CARD
        # ---------------------------------------------------

        st.markdown(
            """
            <div style="
                background:white;
                padding:30px;
                border-radius:24px;
                box-shadow:0 8px 24px rgba(0,0,0,0.06);
                margin-top:20px;
            ">
            """,
            unsafe_allow_html=True
        )

        # Disease Name

        st.markdown(
            f"""
            <h1 style="
                color:#111827;
                font-size:36px;
                font-weight:700;
                margin-bottom:10px;
            ">
                🌱 {disease.replace('___', ' - ')}
            </h1>
            """,
            unsafe_allow_html=True
        )

        # Confidence Score

        st.markdown(
            f"""
            <h3 style="
                color:#166534;
                font-size:24px;
                font-weight:700;
                margin-top:0px;
            ">
                Confidence Score: {round(confidence * 100, 2)}%
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # ---------------------------------------------------
        # DESCRIPTION
        # ---------------------------------------------------

        st.markdown(
            "### Disease Description"
        )

        st.info(
            details["description"]
        )

        # ---------------------------------------------------
        # TREATMENT
        # ---------------------------------------------------

        st.markdown(
            "### Treatment Suggestions"
        )

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

        # ---------------------------------------------------
        # SUSTAINABLE TIPS
        # ---------------------------------------------------

        st.markdown(
            "### Sustainable Care Tips"
        )

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