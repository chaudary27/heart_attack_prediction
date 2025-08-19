
# Heart Attack Prediction Project

This project provides an interactive web app for predicting the risk of heart disease using machine learning. The app is built with Streamlit and uses a trained KNN model.

## Features
- **Interactive Web App:** Enter patient details and get instant risk prediction.
- **Data Preprocessing:** Handles missing values and encodes categorical features automatically.
- **Model Integration:** Uses a pre-trained KNN model and scaler for accurate predictions.
- **User Guidance:** The app checks for missing or invalid inputs and provides clear warnings.
- **Easy to Use:** No coding required—just run the app and use the sliders and dropdowns.

## How to Run
1. Install requirements:
	```sh
	pip install streamlit scikit-learn pandas joblib
	```
2. Start the app:
	```sh
	streamlit run models/app.py
	```
3. Open the provided local URL in your browser.

## Input Features
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST depression)
- ST Slope

## Files
- `models/app.py` — Streamlit app code
- `models/KNN_heart.pkl` — Trained KNN model
- `models/scaler.pkl` — Scaler for preprocessing
- `models/columns.pkl` — Feature columns used during training
- `heart.csv` — Dataset
- `prediction.ipynb` — Data analysis and model training notebook

## Author
Farhan (update with your name if needed)

---

**Note:** If you encounter errors about missing packages, install them using pip as shown above. For feature mismatch errors, ensure you are using the provided model, scaler, and columns files together.
