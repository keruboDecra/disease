import streamlit as st
import time
import numpy as np
from joblib import load

# Suppress warning about invalid feature names
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the best model
best_model = load('best_model.joblib')

# Define feature categories and their respective features
feature_categories = {
    "Loss of Sensory Functions": [
        'loss_of_smell', 'visual_disturbances', 'runny_nose', 'redness_of_eyes', 'sinus_pressure', 
        'dark_urine', 'yellow_urine', 'itching', 'patches_in_throat', 'rusty_sputum'
    ],
    "Joint and Muscle Issues": [
        'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'painful_walking',
        'muscle_pain', 'joint_pain', 'movement_stiffness', 'neck_pain', 'back_pain', 
        'weakness_of_one_body_side', 'pain_in_anal_region'
    ],
    "Psychological Symptoms": [
        'depression', 'irritability', 'lack_of_concentration', 'mood_swings', 'restlessness', 
        'anxiety', 'lethargy', 'excessive_hunger', 'dizziness', 'irregular_sugar_level', 
        'acute_liver_failure', 'coma'
    ],
    "Skin and Nail Issues": [
        'inflammatory_nails', 'brittle_nails', 'skin_peeling', 'skin_rash', 'small_dents_in_nails',
        'yellowish_skin', 'dischromic_patches', 'blister', 'foul_smell_of_urine', 'swollen_blood_vessels'
    ],
    "Respiratory and Nasal Problems": [
        'blood_in_sputum', 'yellow_crust_ooze', 'nodal_skin_eruptions', 'slurred_speech', 'shivering', 
        'sunken_eyes', 'breathlessness', 'cough', 'phlegm', 'congestion', 'fast_heart_rate'
    ],
    "Gastrointestinal Issues": [
        'belly_pain', 'continuous_feel_of_urine', 'nausea', 'stomach_pain', 'loss_of_appetite', 
        'stomach_bleeding', 'bloody_stool', 'dischromic _patches', 'abdominal_pain', 'constipation',
        'diarrhoea', 'bladder_discomfort'
    ],
    "Cardiovascular Symptoms": [
        'chest_pain', 'palpitations', 'history_of_alcohol_consumption', 'high_fever', 
        'red_sore_around_nose', 'yellowing_of_eyes', 'watering_from_eyes', 'swollen_extremeties', 
        'spotting_ urination', 'acute_liver_failure'
    ],
    "Neurological Symptoms": [
        'unsteadiness', 'altered_sensorium', 'spinning_movements', 'weakness_in_limbs', 
        'loss_of_balance', 'visual_disturbances', 'history_of_alcohol_consumption', 
        'toxic_look_(typhos)', 'mucoid_sputum', 'swelling_of_stomach', 'swelled_lymph_nodes'
    ],
    "Other Symptoms": [
        'malaise', 'irritability', 'family_history', 'scurring', 'blackheads', 'sweating', 
        'burning_micturition', 'red_spots_over_body', 'extra_marital_contacts', 'phlegm', 
        'muscle_wasting', 'weight_gain'
    ]
}

# Create a Streamlit app
def main(feature_categories):
    # Define selected_features outside the main function
    selected_features = []
    is_expanded = False
    last_toggled_time = 0

    st.title('Disease Prediction System')

    # Prediction form
    st.subheader('Select Symptoms')
    
    for category, features in feature_categories.items():
        # Add a collapsible section for each category
        st.write(f"### {category}")
        category_checkbox_states = []
        for i, feature in enumerate(features, start=1):
            selected = st.checkbox(feature, key=f"{category}-{i}")
            category_checkbox_states.append(selected)
        
        selected_features.extend([feature for feature, selected in zip(features, category_checkbox_states) if selected])

    # Add slider to adjust threshold
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Adjust the threshold for prediction confidence")

    # Predict disease
    if len(selected_features) >= 4:
        feature_vector = np.zeros(len(feature_categories))
        for feature in selected_features:
            feature_vector[feature_categories.index(feature)] = 1
        
        prediction_proba = best_model.predict_proba([feature_vector])[0]
        prediction = best_model.predict([feature_vector])[0]

        st.subheader("Prediction")
        st.write(f"Predicted Disease: {prediction}")
        st.write(f"Confidence: {prediction_proba.max() * 100:.2f}%")

        # Add a recommendation for more symptoms if less than 4 are selected
    else:
        st.warning("Please select at least 4 symptoms for accurate prediction.")

    # Clear input button
    if st.button("Clear Input"):
        selected_features.clear()

    # Collapse the expanders after 10 seconds
    if time.time() - last_toggled_time > 10:
        is_expanded = False

if __name__ == '__main__':
    main(feature_categories)
