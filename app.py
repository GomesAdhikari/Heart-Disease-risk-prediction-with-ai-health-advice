import os
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, flash
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# Set up Google Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = joblib.load('best_rf_model.pkl')

# Initialize the Google Gemini model
model_llm = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Define the prompt template
template = '''
Given the following patient information:

- Age: {age}
- Sex: {sex}
- Chest Pain Type: {chest_pain_type}
- Resting Blood Pressure: {resting_bp}
- Cholesterol Level: {cholesterol}
- Fasting Blood Sugar: {fasting_bs}
- Resting ECG: {resting_ecg}
- Maximum Heart Rate: {max_hr}
- Exercise Angina: {exercise_angina}
- Oldpeak: {oldpeak}
- ST Slope: {st_slope}

Please provide detailed recommendations in the following format:

1. Dietary Changes
- Foods to Include: Specify the types of foods that should be included to manage cholesterol levels and improve heart health.
- Foods to Avoid: List foods that should be avoided to prevent high cholesterol and cardiovascular issues.

2. Exercise Routines
- Types of Exercises: Recommend types of exercises that are beneficial for the patient.
- Frequency and Intensity: Indicate how often and at what intensity the patient should engage in these exercises.

3. Stress Management
- Techniques: Suggest effective techniques for managing and reducing stress.

4. Interpretation of Readings
- Ideal or Average Values: Provide the ideal or average values for each health parameter (e.g., cholesterol levels, blood pressure).
- Importance: Explain why maintaining these readings is crucial for cardiovascular health.

5. Action Plan for High Readings
- High Readings: Offer specific steps to address high readings or high cholesterol.
- Lifestyle Changes: Recommend lifestyle changes to improve readings.
- Follow-Up Actions: Suggest any follow-up actions needed and the potential need for medical consultation.

6. Medications for the patient according to the readings.
- Medication for cholesterol if need (according to the readings)
- Medication for Blood Pressure if need (according to the readings)
- Medication for Old Peak depression if need (according to the readings)
- Medication for FastingBloodSugar if need (according to the readings)
'''
app = Flask(__name__)
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    chest_pain_type = request.form.get('ChestPainType')

    # Initialize all chest pain type fields to 0
    chest_pain_type_ata = 0
    chest_pain_type_nap = 0
    chest_pain_type_ta = 0

    # Set the correct value based on the selected chest pain type
    if chest_pain_type == '1':
        chest_pain_type_ata = 1
    elif chest_pain_type == '2':
        chest_pain_type_nap = 1
    elif chest_pain_type == '3':
        chest_pain_type_ta = 1

    features = [
        float(request.form['Age']),
        float(request.form['Sex']),
        float(request.form['RestingBP']),
        float(request.form['Cholesterol']),
        float(request.form['FastingBS']),
        float(request.form['RestingECG']),
        float(request.form['MaxHR']),
        float(request.form['ExerciseAngina']),
        float(request.form['Oldpeak']),
        float(request.form['ST_Slope']),
        chest_pain_type_ata,
        chest_pain_type_nap,
        chest_pain_type_ta
    ]

    features = np.array(features).reshape(1, -1)
    prediction_proba = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]
    if prediction == 0:
        prediction = 'Low chance of Heart Disease.'
    else:
        prediction = 'High chance of heart disease.'

    # Format probabilities for display
    proba_heart_disease = prediction_proba[1] * 100
    proba_no_heart_disease = prediction_proba[0] * 100

    # Provide advice based on predictions using the LLM
    prompt = template.format(
        age=request.form['Age'],
        sex=request.form['Sex'],
        chest_pain_type=request.form['ChestPainType'],
        resting_bp=request.form['RestingBP'],
        cholesterol=request.form['Cholesterol'],
        fasting_bs=request.form['FastingBS'],
        resting_ecg=request.form['RestingECG'],
        max_hr=request.form['MaxHR'],
        exercise_angina=request.form['ExerciseAngina'],
        oldpeak=request.form['Oldpeak'],
        st_slope=request.form['ST_Slope']
    )

    response = model_llm.generate_content([prompt])
    rep = response.text.replace('*',"-")

    return render_template('result.html',
                           prediction=prediction,
                           proba_heart_disease=proba_heart_disease,
                           proba_no_heart_disease=proba_no_heart_disease,
                           response=rep)
    

if __name__ == '__main__':
    app.run(debug=True)
