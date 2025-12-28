import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */ /* Keep this commented out so you can see the sidebar arrow */
    
    /* Custom Font */
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Metrics Styling - FIXED COLOR */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6; /* Light gray background */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* Force text inside metrics to be BLACK so it shows on the light card */
    div[data-testid="stMetric"] > div {
        color: #000000 !important;
    }
    
    /* Target the Label (e.g., "Default Probability") specifically */
    div[data-testid="stMetric"] label {
        color: #31333F !important; /* Dark Gray */
    }
    
    /* Target the Value (e.g., "43.30%") specifically */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Pitch Black */
    }

    /* Button Styling */
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = jb.load("models/logistic_regression_model.pkl")
        scaler = jb.load("models/scaler.pkl")
        features = jb.load("models/model_features.pkl")
        return model, scaler, features
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please check your 'models/' directory.")
        return None, None, None

model, scaler, features = load_assets()

with st.sidebar:
    st.title("💳 Configuration")
    st.write("Adjust customer details below.")
    
    with st.expander("👤 Demographics", expanded=True):
        sex_label = st.selectbox("Sex", ["Male", "Female"])
        SEX = 1 if sex_label == "Male" else 2
        
        education_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
        education_label = st.selectbox("Education Level", list(education_map.keys()))
        EDUCATION = education_map[education_label]
        
        marriage_map = {"Married": 1, "Single": 2, "Others": 3}
        marriage_label = st.selectbox("Marital Status", list(marriage_map.keys()))
        MARRIAGE = marriage_map[marriage_label]
        
        AGE = st.slider("Age", 18, 80, 35)

    with st.expander("💰 Financials", expanded=False):
        LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=0, value=200000, step=10000)
        avg_bill_amt = st.number_input("Avg Bill Amount (6 mo)", min_value=0, value=50000, step=5000)
        avg_pay_amt = st.number_input("Avg Payment Amount (6 mo)", min_value=0, value=40000, step=5000)

    with st.expander("📉 Behavior", expanded=False):
        late_payment_count = st.slider("Late Payments (count)", 0, 6, 1)
        max_delay = st.slider("Max Delay (months)", 0, 10, 1)

   
    log_limit_bal = np.log1p(LIMIT_BAL)
    log_bill_amt_avg = np.log1p(avg_bill_amt)
    log_pay_amt_avg = np.log1p(avg_pay_amt)

    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Risk Profile")

st.title("Credit Default Risk Predictor")
st.markdown("Machine Learning–based assessment system.")

if predict_btn and model is not None:
   
    input_df = pd.DataFrame([[SEX, EDUCATION, MARRIAGE, AGE, log_limit_bal, log_bill_amt_avg, log_pay_amt_avg, late_payment_count, max_delay]], columns=features)
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

   
    st.markdown("### 📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Default Probability", value=f"{prob:.2%}")
    
    with col2:
        payment_ratio = avg_pay_amt / (avg_bill_amt + 1)
        st.metric(label="Payment-to-Bill Ratio", value=f"{payment_ratio:.2f}")

    st.divider()

    if prob < 0.30:
        st.success("✅ **Low Risk**")
        risk_color = "green"
    elif prob < 0.60:
        st.warning("⚠️ **Medium Risk**")
        risk_color = "red"
    else:
        st.error("🛑 **High Risk**")
        risk_color = "red"

    st.divider()

    
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Payment vs. Bill History")
        bill_pay_df = pd.DataFrame({
            "Amount": [avg_bill_amt, avg_pay_amt],
            "Type": ["Bill Amount", "Payment Amount"]
        }).set_index("Type")
        st.bar_chart(bill_pay_df, color=["#F9F5F58B"])

    with c2:
        st.subheader("Risk Factors")    
        risk_data = pd.DataFrame({
            "Factor": ["Late Frequency", "Delay Severity", "Probability"],
            "Score": [late_payment_count/6, max_delay/10, prob]
        }).set_index("Factor")
        st.bar_chart(risk_data, color=["#F9F5F588"])


    st.divider()
    with st.expander("📝 View Detailed Summary Report"):
        st.write(f"""
        **Customer Profile:**
        - **Age:** {AGE} | **Status:** {marriage_label} | **Education:** {education_label}
        - **Credit Limit:** ${LIMIT_BAL:,}
        
        **Behavioral Flags:**
        - Has delayed payment **{max_delay} months** maximum.
        - Late payments occurred **{late_payment_count} times** in the last 6 months.
        """)
elif not predict_btn:
    # Spacer to push content down slightly
    st.write("")
    st.write("")

    # Centered Welcome Message
    st.markdown("""
    <div style="text-align: center;">
        <h1>👋 Welcome to Credit Risk AI</h1>
        <p style="font-size: 1.2rem;">
            This tool uses <strong>Machine Learning</strong> to assess customer default probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 3-Column Feature Highlight
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🛡️ Reliable")
        st.write("Trained on thousands of historical credit records for accuracy.")

    with col2:
        st.markdown("### ⚡ Instant")
        st.write("Get real-time risk predictions and probability scores.")

    with col3:
        st.markdown("### 📊 Visual")
        st.write("Understand the 'Why' behind the risk with clear charts.")
    
    # Simple instruction at the bottom
    st.info("👈 **To get started:** Open the Sidebar, adjust the settings, and click 'Analyze Risk Profile'.")