import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")

# Continuous features for scaling
continuous_features = ['UPDRS', 'MoCA', 'FunctionalAssessment', 'Age']

st.title("Parkinson's Disease Prediction App")

st.write("Please enter the following patient information:")

# Take user input
Tremor = st.number_input("Tremor (0 or 1)", min_value=0, max_value=1)
Rigidity = st.number_input("Rigidity (0 or 1)", min_value=0, max_value=1)
Bradykinesia = st.number_input("Bradykinesia (0 or 1)", min_value=0, max_value=1)
PosturalInstability = st.number_input("PosturalInstability (0 or 1)", min_value=0, max_value=1)
UPDRS = st.number_input("UPDRS", value=0.0)
MoCA = st.number_input("MoCA", value=0.0)
FunctionalAssessment = st.number_input("FunctionalAssessment", value=0.0)
SpeechProblems = st.number_input("SpeechProblems (0 or 1)", min_value=0, max_value=1)
SleepDisorders = st.number_input("SleepDisorders (0 or 1)", min_value=0, max_value=1)
Constipation = st.number_input("Constipation (0 or 1)", min_value=0, max_value=1)
Age = st.number_input("Age", value=0.0)
FamilyHistoryParkinsons = st.number_input("FamilyHistoryParkinsons (0 or 1)", min_value=0, max_value=1)

# Predict button
if st.button("Predict"):
    # Prepare input dataframe
    input_df = pd.DataFrame([[
        Tremor, Rigidity, Bradykinesia, PosturalInstability,
        UPDRS, MoCA, FunctionalAssessment,
        SpeechProblems, SleepDisorders, Constipation,
        Age, FamilyHistoryParkinsons
    ]], columns=[
        'Tremor','Rigidity','Bradykinesia','PosturalInstability',
        'UPDRS','MoCA','FunctionalAssessment',
        'SpeechProblems','SleepDisorders','Constipation',
        'Age','FamilyHistoryParkinsons'
    ])
    
    # Scale continuous features
    input_df[continuous_features] = scaler.transform(input_df[continuous_features])
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display result
    if prediction == 1:
        st.error("Prediction: Parkinson's Disease")
    else:
        st.success("Prediction: Healthy")
    
    st.write(f"Probability Healthy: {prediction_proba[0]:.2f}")
    st.write(f"Probability Parkinsons: {prediction_proba[1]:.2f}")
