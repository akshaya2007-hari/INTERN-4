import streamlit as st
import pickle
import numpy as np
from sklearn.exceptions import NotFittedError

# Page config
st.set_page_config(page_title="Late Delivery Prediction")
st.title("🚚 Late Delivery Prediction App")

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Upload late_model.pkl to GitHub.")
    st.stop()

st.write("Enter delivery details:")

# User inputs
distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    step=0.1
)

traffic = st.number_input(
    "Traffic level (e.g., 1 = Low, 2 = Medium, 3 = High)",
    min_value=1,
    max_value=3,
    step=1
)

# Predict button
if st.button("Predict Late Status"):
    try:
        input_data = np.array([[distance, traffic]])

        # Class prediction
        prediction = model.predict(input_data)[0]

        # Probability prediction
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⏰ Delivery will be LATE (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Delivery will be ON TIME (Probability: {1 - probability:.2f})")

    except NotFittedError:
        st.error("❌ Model is not trained. Please retrain and upload a fitted model.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
