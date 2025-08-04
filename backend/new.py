import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
# Import your model and preprocessing functions

# Page configuration
st.set_page_config(
    page_title="Wildfire Risk Prediction",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("🔥 Wildfire Risk Prediction System")
st.markdown("Predict wildfire risk based on environmental and weather conditions")

# Sidebar for input features
st.sidebar.header("Input Parameters")

# Example input features (adjust based on your model)
temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 10)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 100, 0)
vegetation_density = st.sidebar.slider("Vegetation Density", 0.0, 1.0, 0.5)

# Load your trained model (adjust path as needed)
@st.cache_resource
def load_model():
    try:
        # Try different possible model file names
        model = joblib.load('model.pkl')  # or pickle.load()
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure your trained model is saved as 'model.pkl'")
        return None

# Prediction function
def predict_wildfire_risk(features):
    model = load_model()
    if model is not None:
        # Reshape features for prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)
        return prediction[0], probability[0]
    return None, None

# Main prediction section
st.header("Prediction Results")

if st.button("Predict Wildfire Risk", type="primary"):
    # Collect all input features
    features = [temperature, humidity, wind_speed, rainfall, vegetation_density]
    
    prediction, probability = predict_wildfire_risk(features)
    
    if prediction is not None:
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Level")
            risk_levels = ["Low", "Medium", "High"]  # Adjust based on your model
            risk_level = risk_levels[int(prediction)]
            
            # Color code the risk level
            if risk_level == "Low":
                st.success(f"🟢 {risk_level} Risk")
            elif risk_level == "Medium":
                st.warning(f"🟡 {risk_level} Risk")
            else:
                st.error(f"🔴 {risk_level} Risk")
        
        with col2:
            st.subheader("Prediction Confidence")
            max_prob = max(probability)
            st.metric("Confidence", f"{max_prob:.2%}")
    
    # Display input summary
    st.subheader("Input Summary")
    input_data = {
        "Parameter": ["Temperature", "Humidity", "Wind Speed", "Rainfall", "Vegetation Density"],
        "Value": [f"{temperature}°C", f"{humidity}%", f"{wind_speed} km/h", f"{rainfall} mm", f"{vegetation_density:.2f}"]
    }
    st.table(pd.DataFrame(input_data))

# Additional information
st.markdown("---")
st.markdown("### About This Model")
st.info("This wildfire risk prediction model uses environmental and weather data to assess the likelihood of wildfire occurrence. The model considers factors such as temperature, humidity, wind speed, rainfall, and vegetation density.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")