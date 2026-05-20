@st.cache_resource
def load_ml_model():

    return load_model(
        "model/plant_disease_model.h5",
        compile=False,
        safe_mode=False
    )