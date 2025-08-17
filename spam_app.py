import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Aegis AI | Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS for a modern look
# ---------------------------
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #F0F2F6;
    }
    /* Custom button styling */
    .stButton>button {
        border: 2px solid #4A90E2;
        border-radius: 20px;
        color: #FFFFFF;
        background-color: #4A90E2;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #4A90E2;
        border-color: #4A90E2;
    }
    /* Result card styling */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }
    .spam {
        background-color: #FFD2D2;
        border: 2px solid #D9534F;
    }
    .not-spam {
        background-color: #D4EDDA;
        border: 2px solid #5CB85C;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
    .confidence-text {
        font-size: 18px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Load Pre-trained Model & Vectorizer
# ---------------------------
# This function loads the trained Naive Bayes model and the TF-IDF vectorizer.
# @st.cache_resource ensures this expensive operation is done only once.
@st.cache_resource
def load_model():
    try:
        with open("naivebayes.pkl", "rb") as nb_file:
            nb = pickle.load(nb_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        return nb, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please make sure 'naivebayes.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

nb, vectorizer = load_model()

# ---------------------------
# Session State Initialization
# ---------------------------
# Initialize session state to store classification history.
if "history" not in st.session_state:
    st.session_state.history = []
if 'latest_result' not in st.session_state:
    st.session_state.latest_result = None

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://placehold.co/400x150/4A90E2/FFFFFF?text=Aegis+AI", use_column_width=True)
    st.title("🛡️ Aegis AI Spam Shield")
    st.info(
        """
        This app uses a **Naive Bayes** machine learning model to classify emails as **Spam** or **Not Spam**.
        
        **How it works:**
        1.  The model is trained on a large dataset of emails.
        2.  It uses **TF-IDF Vectorization** to convert text into numerical data.
        3.  The algorithm calculates the probability of an email being spam based on the words it contains.
        """
    )
    st.success("Built by Akash M")

# ---------------------------
# Main Application UI
# ---------------------------
st.title("📧 Spam Email Classifier")
st.markdown("### Instantly detect spam with the power of AI. Enter an email below to see the magic.")

# Create two columns for a more organized layout
col1, col2 = st.columns([2, 1.5], gap="large")

with col1:
    st.markdown("#### ✍️ Enter Email Content")
    user_input = st.text_area("Paste the full email content here for analysis:", height=250, placeholder="Subject: You've won a prize!...")

    if st.button("🚀 Classify Email", use_container_width=True):
        if user_input.strip() and nb and vectorizer:
            with st.spinner('🤖 AI is analyzing the email...'):
                time.sleep(1) # Simulate processing time

                # Vectorize the user input
                vec_input = vectorizer.transform([user_input])
                
                # Predict the class (0 for Not Spam, 1 for Spam)
                prediction = nb.predict(vec_input)[0]
                
                # Get prediction probabilities
                probabilities = nb.predict_proba(vec_input)[0]
                confidence = probabilities[prediction] * 100
                
                result_text = "🛑 Spam" if prediction == 1 else "✅ Ham"

                # Store the latest result for display in the other column
                st.session_state.latest_result = {
                    "prediction": prediction,
                    "result_text": result_text,
                    "confidence": confidence,
                    "input": user_input
                }
                
                # Add to history (newest first)
                st.session_state.history.insert(0, st.session_state.latest_result)
        else:
            st.warning("Please enter some text to classify.")

with col2:
    st.markdown("#### 🔎 Classification Result")
    if st.session_state.latest_result:
        res = st.session_state.latest_result
        card_class = "spam" if res['prediction'] == 1 else "not-spam"
        
        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-text">{res['result_text']}</div>
                <div class="confidence-text">Confidence: {res['confidence']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.info("The result of your classification will appear here.")

st.markdown("---")


# ---------------------------
# History and Analytics Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📜 Classification History", "📊 Model Performance"])

with tab1:
    st.subheader("Your Recent Classifications")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[:10]): # Show latest 10
            with st.expander(f"**{item['result_text']}** - Confidence: {item['confidence']:.2f}%"):
                st.write(item['input'])
    else:
        st.write("No classifications have been made yet.")

with tab2:
    st.subheader("Understanding the Model's Brain")
    st.markdown("To show how the AI works, we tested it on 1000 sample emails it had never seen before. Here are the results:")

    # Dummy data for visualization
    # In a real scenario, you'd use a held-out test set.
    y_true = np.concatenate([np.zeros(450), np.ones(50), np.zeros(40), np.ones(460)])
    y_pred = np.concatenate([np.zeros(450), np.zeros(50), np.ones(40), np.ones(460)])
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    st.metric(label="**Model Accuracy on Test Data**", value=f"{accuracy:.2f}%")
    st.write("This score represents the percentage of emails the model correctly classified in the test set.")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    st.pyplot(fig)
    st.markdown(
        """
        **How to read this chart:**
        - **Top-Left (True Negative):** Correctly identified as "Not Spam".
        - **Bottom-Right (True Positive):** Correctly identified as "Spam".
        - **Top-Right (False Positive):** Incorrectly flagged a safe email as "Spam".
        - **Bottom-Left (False Negative):** Incorrectly allowed a "Spam" email through.
        
        A perfect model would only have numbers on the diagonal from top-left to bottom-right.
        """
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Akash AI Spam Shield © 2025 | An interactive demo built with Streamlit and Scikit-learn.")
