import streamlit as st
import pandas as pd
from load import load_models

models, meta = load_models()

scaler = meta['scaler']
eigenvectors = meta['eigenvectors']

mapping = {
    1: 'Malignant',
    0: 'Benign'
}
def check_columns(columns: list):
    required_cols = sorted(['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'])
    
    not_in = []
    for col in required_cols:
        if col not in columns:
            not_in.append(col)
    
    return not_in

st.title("Breast Cancer Classification")
model_choice = st.selectbox("Select a model:", models.keys())

uploaded_file = st.file_uploader("Choose a CSV file containing diagnostic features", type="csv")

if uploaded_file is not None:
    # Read as pandas dataframe
    df = pd.read_csv(uploaded_file)
    col_check = check_columns(df.columns)
    if (len(col_check) > 0):
        st.error(f"CSV file missing the column(s): {', '.join(col_check)}")
    else:
        df.index = [f"test {i+1}" for i in range(len(df))]
        
        st.dataframe(df)
        
        model = models[model_choice]
        
        scaled = scaler.transform(df)
        projected = scaled @ eigenvectors
        predictions = model.predict(projected)
        
        for i, prediction in enumerate(predictions):
            st.markdown(f"### Prediction for test {i + 1}")
            if prediction == 0:
                st.success("✅ Prediction: Benign Cell")
            else:
                st.error("⚠️ Prediction: Cancerous Cell Detected. Get in touch with a medical professional as soon as possible!")
            
            st.markdown('---')
        
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 14px;

        background-color: var(--background-color);
        color: var(--text-color);
        border-top: 1px solid var(--secondary-background-color);
    }
    </style>

    <div class="footer">
        © 2025 Ananth | Made with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

    