import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Wildfire Risk Prediction",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("🔥 Wildfire Risk Prediction System")
st.markdown("Predict wildfire risk based on environmental conditions")

# Load model and preprocessor
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("knn_model.pkl")
        scaler = joblib.load("knn_scaler.pkl")
        selected_features = joblib.load("selected_features.pkl")
        st.success("✅ Model components loaded successfully!")
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"❌ Error loading model components: {str(e)}")
        return None, None, None

# Load components
model, scaler, selected_features = load_model_components()

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 About", "📈 Indicators", "🗺️ Data"])

with tab1:
    st.header("Wildfire Risk Prediction")
    
    if model is not None:
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Parameters")
            dc = st.number_input(
                "DC (Drought Code)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=100.0,
                help="Drought Code - indicates moisture content of deep organic layers"
            )
            
            temp = st.number_input(
                "Temperature (°C)", 
                min_value=-50.0, 
                max_value=60.0, 
                value=20.0,
                help="Temperature in Celsius"
            )
        
        with col2:
            st.subheader("Month Indicators")
            month_dec = st.selectbox(
                "December (1 if December, 0 otherwise)", 
                options=[0, 1],
                help="Select 1 if the month is December"
            )
            
            month_mar = st.selectbox(
                "March (1 if March, 0 otherwise)", 
                options=[0, 1],
                help="Select 1 if the month is March"
            )
        
        # Optional location inputs
        st.subheader("Location (Optional)")
        col3, col4 = st.columns(2)
        with col3:
            latitude = st.number_input("Latitude", value=0.0)
        with col4:
            longitude = st.number_input("Longitude", value=0.0)
        
        # Prediction button
        if st.button("🔥 Predict Fire Risk", type="primary"):
            try:
                # Prepare input data
                input_values = np.array([dc, temp, month_dec, month_mar]).reshape(1, -1)
                
                # Scale the input
                scaled_input = scaler.transform(input_values)
                
                # Make prediction
                prediction = int(model.predict(scaled_input)[0])
                
                # Get prediction probability if available
                try:
                    prediction_proba = model.predict_proba(scaled_input)[0]
                    confidence = max(prediction_proba)
                except:
                    confidence = None
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if prediction == 1:
                        st.error("⚠️ **HIGH FIRE RISK DETECTED!**")
                        st.markdown("🔴 **Risk Level: HIGH**")
                    else:
                        st.success("✅ **LOW FIRE RISK**") 
                        st.markdown("🟢 **Risk Level: LOW**")
                
                with col_result2:
                    if confidence:
                        st.metric("Prediction Confidence", f"{confidence:.1%}")
                    st.metric("Risk Score", prediction)
                
                # Display input summary
                st.subheader("📋 Input Summary")
                summary_data = {
                    "Parameter": ["Drought Code (DC)", "Temperature", "December", "March", "Latitude", "Longitude"],
                    "Value": [f"{dc:.1f}", f"{temp:.1f}°C", month_dec, month_mar, f"{latitude:.2f}", f"{longitude:.2f}"]
                }
                st.table(pd.DataFrame(summary_data))
                
                # Save to session state for potential database storage
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = []
                
                prediction_data = {
                    'dc': dc,
                    'temp': temp,
                    'month_dec': month_dec,
                    'month_mar': month_mar,
                    'prediction': prediction,
                    'latitude': latitude,
                    'longitude': longitude,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.predictions.append(prediction_data)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    else:
        st.error("❌ Cannot make predictions - model not loaded properly")

with tab2:
    st.header("📊 About the Model")
    st.markdown("""
    ### Wildfire Risk Prediction System
    
    This system uses machine learning to predict wildfire risk based on environmental conditions.
    
    **Key Features:**
    - **Algorithm**: K-Nearest Neighbors (KNN)
    - **Input Features**: Drought Code, Temperature, Month indicators
    - **Output**: Binary classification (High Risk / Low Risk)
    
    **Model Performance:**
    - Trained on historical wildfire data
    - Uses feature scaling for optimal performance
    - Incorporates seasonal patterns through month indicators
    
    **How it works:**
    1. Input environmental parameters
    2. Data is scaled using the same preprocessing as training
    3. KNN algorithm finds similar historical patterns
    4. Returns risk prediction with confidence score
    """)

with tab3:
    st.header("📈 Fire Weather Indicators")
    st.markdown("""
    ### Understanding Fire Weather Parameters
    
    **Drought Code (DC):**
    - Indicates moisture content of deep organic layers
    - Higher values = drier conditions = higher fire risk
    - Range: 0-1000+ (typical range 0-800)
    
    **Temperature:**
    - Air temperature in Celsius
    - Higher temperatures increase fire risk
    - Combined with other factors for comprehensive assessment
    
    **Seasonal Factors:**
    - **March**: Spring fire season in many regions
    - **December**: Winter conditions, generally lower risk but varies by location
    
    **Risk Levels:**
    - 🟢 **Low Risk (0)**: Favorable conditions, low fire probability
    - 🔴 **High Risk (1)**: Dangerous conditions, high fire probability
    """)

with tab4:
    st.header("🗺️ Prediction History")
    
    if 'predictions' in st.session_state and st.session_state.predictions:
        st.subheader("Recent Predictions")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.predictions)
        
        # Display as table
        st.dataframe(df, use_container_width=True)
        
        # Simple visualization
        if len(df) > 1:
            st.subheader("📊 Risk Distribution")
            risk_counts = df['prediction'].value_counts()
            risk_labels = {0: 'Low Risk', 1: 'High Risk'}
            risk_counts.index = risk_counts.index.map(risk_labels)
            st.bar_chart(risk_counts)
    else:
        st.info("No predictions made yet. Use the Prediction tab to make your first prediction!")
        
    # Clear history button
    if st.button("🗑️ Clear Prediction History"):
        st.session_state.predictions = []
        st.success("Prediction history cleared!")
        st.rerun()

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 Wildfire Prediction")
st.sidebar.markdown("Built with Streamlit")
st.sidebar.markdown("Machine Learning Model: KNN")

if model is not None:
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.error("❌ Model Error")