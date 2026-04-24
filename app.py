import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Configuration
MODEL_PATH = r"psoriasis_severity_model.h5"
CLASSES = ['mild', 'moderate', 'severe']

# Page Config
st.set_page_config(
    page_title="Psoriasis Severity AI",
    page_icon="🩺",
    layout="wide"
)

# Load Model (Cached)
@st.cache_resource
def load_learner():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_learner()

# --- Logic ---
def predict_image(image, model):
    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    
    return CLASSES[class_idx], confidence

def get_advice(severity):
    advice = {
        "mild": {
            "care": [
                "Regular moisturization (Emollients).",
                "Apply mild medicated creams (non-prescription).",
                "Avoid skin dryness.",
                "Short sun exposure (10-15 mins) can help.",
                "Use gentle, fragrance-free soaps."
            ],
            "action": "Monitor and maintain skin hydration. Stress control."
        },
        "moderate": {
            "care": [
                "Consider consulting a Dermatologist.",
                "Medicated topical creams (Vitamin D analogues, corticosteroids).",
                "Controlled sun exposure (avoid burning).",
                "Identify and avoid triggers (Stress, Smoking).",
                "Keep skin clean and well-moisturized."
            ],
            "action": "Dermatologist consultation recommended."
        },
        "severe": {
            "care": [
                "URGENT: Consult a Dermatologist immediately.",
                "May require systemic treatment or phototherapy.",
                "Prevent infection in cracked skin areas.",
                "Intensive skin hydration regimen.",
                "Significant lifestyle modifications required."
            ],
            "action": "Seek professional medical help immediately."
        }
    }
    return advice.get(severity, {})

# --- UI Layout ---

# --- UI Layout ---

# --- UI Layout ---

# Custom CSS for aesthetics (Dark Mode Friendly)
st.markdown("""
<style>
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Card-like styling that adapts to theme (using semi-transparent background) */
    .st-emotion-cache-1y4p8pa {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.05); /* Subtle overlay */
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Severity Badges */
    .severity-mild {
        background-color: rgba(40, 167, 69, 0.2);
        color: #2ecc71;
        border: 1px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .severity-moderate {
        background-color: rgba(255, 193, 7, 0.2);
        color: #f1c40f;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .severity-severe {
        background-color: rgba(220, 53, 69, 0.2);
        color: #e74c3c;
        border: 1px solid #dc3545;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("MediSkin AI")
    st.markdown("### Psoriasis Assessment")
    st.info("AI-powered severity grading & care recommendations.")
    st.markdown("---")
    with st.expander("ℹ️ Disclaimer"):
        st.warning("Educational tool only. Not a medical diagnosis. Consult a dermatologist.")

# Main Header
st.title("🩺 Psoriasis Severity & Care AI")
st.markdown("Upload a skin lesion image for instant analysis.")
st.markdown("---")

# Main Content Grid
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Select skin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Patient Image', use_container_width=True)

with col2:
    st.subheader("2. Analysis Results")
    
    if uploaded_file is not None and model:
        with st.spinner('Analyzing lesions...'):
            severity, confidence = predict_image(image, model)
        
        # CSS Class selection
        css_class = f"severity-{severity}"
        
        # Display Severity Badge
        st.markdown(f"""
        <div class='{css_class}'>
            <h2 style='margin:0; padding:0;'>{severity.upper()}</h2>
            <p style='margin:0; opacity: 0.8;'>Detected Severity Level</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        # Confidence Score with Progress Bar
        st.write(f"**AI Confidence Score:** {confidence:.1%}")
        st.progress(float(confidence))
        
        if confidence < 0.60:
            st.caption("⚠️ Note: The confidence is moderate. Ensure image is clear and focused on the lesion.")

        st.markdown("---")
        st.subheader("3. Recommendations")
        
        advice = get_advice(severity)
        
        # Tabs for Advice
        tab_care, tab_diet, tab_safe = st.tabs(["💊 Treatment", "🥗 Nutrition", "🛡️ Lifestyle"])
        
        with tab_care:
            st.success(f"**Action**: {advice['action']}")
            for item in advice['care']:
                st.markdown(f"- {item}")
                
        with tab_diet:
            st.markdown("**✅ Recommended:**")
            st.markdown("- Omega-3 (Fish, Flax)")
            st.markdown("- Anti-inflammatory Berries & Greens")
            st.markdown("**❌ Avoid:**")
            st.markdown("- Alcohol & Smoking")
            st.markdown("- Excess Sugar & Processed Food")
            
        with tab_safe:
            st.warning("**Precautions:**")
            st.markdown("- Do NOT scratch lesions")
            st.markdown("- Use lukewarm water for bathing")
            st.markdown("- Wear soft cotton clothing")

    elif uploaded_file is None:
        st.info("👈 Waiting for image upload...")
    elif not model:
        st.error("Model failed to load. Please check server logs.")
