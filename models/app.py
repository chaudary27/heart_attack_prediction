import streamlit as st
import pandas as pd
import joblib


model=joblib.load(r'E:\Githubb\git_repo_ML\heart_attack_prediction\models\KNN_heart.pkl')
scaler=joblib.load(r'E:\Githubb\git_repo_ML\heart_attack_prediction\models\scaler.pkl')
expected_columns=joblib.load(r'E:\Githubb\git_repo_ML\heart_attack_prediction\models\columns.pkl')

st.title('Heart strock prediction (by farhan)')
st.markdown('provide the following details')

age=st.slider('age',18,100,40)
sex=st.selectbox('Sex',['M','F'])
chest_pain=st.selectbox('Chest Pain Type ',['ATA','NAP','TA','ASY'])
resting_bp=st.number_input('Resting blood pressure(mmHG)' , 80,200,120)
chlestrol=st.number_input('Chlestrol (mg/dL)',100,600,200)
# ...existing code...

fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['Y', 'N'])
oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 10.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# Prepare input as DataFrame
input_dict = {
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [chlestrol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
}

input_df = pd.DataFrame(input_dict)

# Check for missing values in user input
missing = False
for col in input_df.columns:
    if pd.isnull(input_df[col][0]) or input_df[col][0] == '':
        st.warning(f"Please provide a valid value for {col}.")
        missing = True

if not missing:
    # One-hot encode user input
    input_encoded = pd.get_dummies(input_df)
    # Reindex to match expected columns (order and names), fill missing with 0
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    # Scale the entire input_encoded DataFrame (scaler expects all columns)
    input_encoded = pd.DataFrame(scaler.transform(input_encoded), columns=input_encoded.columns)

    if st.button('Predict'):
        prediction = model.predict(input_encoded)[0]
        if prediction == 1:
            st.error('High risk of heart disease detected.')
        else:
            st.success('Low risk of heart disease detected.')









