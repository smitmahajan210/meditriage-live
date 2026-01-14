import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIGURATION ---
# This is where your Backend API is running
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MediTriage AI", layout="wide")

# --- HEADER ---
st.title("🏥 MediTriage: AI-Powered Hospital Assistant")
st.markdown("""
This system uses **Deep Learning** to assist hospital staff:
1.  **Digitize** handwritten patient IDs using CNNs (VGG13).
2.  **Triage** patients by predicting health risk using Neural Networks (MLP).
""")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Patient Risk Assessment", "📝 Handwriting OCR"])

# --- TAB 1: RISK PREDICTION ---
with tab1:
    st.header("Patient Vitals Entry")
    st.write("Enter the patient's 7 key bio-markers to predict health risk.")

    col1, col2 = st.columns(2)
    
    with col1:
        f1 = st.number_input("Feature 1 (e.g., Glucose)", value=0.0)
        f2 = st.number_input("Feature 2 (e.g., Blood Pressure)", value=0.0)
        f3 = st.number_input("Feature 3 (e.g., Skin Thickness)", value=0.0)
        f4 = st.number_input("Feature 4 (e.g., Insulin)", value=0.0)
    with col2:
        f5 = st.number_input("Feature 5 (e.g., BMI)", value=0.0)
        f6 = st.number_input("Feature 6 (e.g., Pedigree Function)", value=0.0)
        f7 = st.number_input("Feature 7 (e.g., Age)", value=0.0)

    if st.button("Analyze Risk Profile"):
        payload = {
            "f1": f1, "f2": f2, "f3": f3, "f4": f4, 
            "f5": f5, "f6": f6, "f7": f7
        }
        
        try:
            with st.spinner("Consulting AI Model..."):
                response = requests.post(f"{API_URL}/predict-risk", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                risk_prob = result["risk_probability"]
                prediction = result["prediction"]
                
                if prediction == "High Risk":
                    st.error(f"⚠️ **Prediction: HIGH RISK** (Probability: {risk_prob:.2%})")
                else:
                    st.success(f"✅ **Prediction: Low Risk** (Probability: {risk_prob:.2%})")
            else:
                st.error("Error communicating with backend.")
        except Exception as e:
            st.error(f"Connection Error: {e}. Is the backend running?")

# --- TAB 2: HANDWRITING OCR ---
with tab2:
    st.header("Digitize Patient Records")
    st.write("Upload an image of a handwritten character (e.g., from a Patient ID form).")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        
        if st.button("Read Handwriting"):
            try:
                # Prepare file for API
                # Reset pointer to beginning of file
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                
                with st.spinner("Scanning Image..."):
                    response = requests.post(f"{API_URL}/read-id", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    char_detected = result["detected_id_char"]
                    st.success(f"🧠 AI Recognized Character: **{char_detected}**")
                else:
                    st.error("Error reading image.")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- SIDEBAR INFO ---
st.sidebar.title("System Info")
st.sidebar.info("Model Status: Online ✅")
st.sidebar.markdown("---")
st.sidebar.text("Built with PyTorch, FastAPI, & Streamlit")