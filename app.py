import streamlit as st
import numpy as np
from joblib import load
from streamlit.hashing import _CodeHasher
from streamlit.server.server import Server
import json

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

# Function to get the session state
def get_session():
    session = Server.get_current()._get_session_info_hash()
    return session

# Create a Streamlit app
def main():
    session_id = get_session()

    # Define selected_features in session state
    if 'selected_features' not in session_id:
        session_id['selected_features'] = []

    st.title('Disease Prediction System')

    # Sidebar for adjusting threshold and clearing input
    threshold = st.sidebar.slider('Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if st.sidebar.button('Clear Input'):
        session_id['selected_features'] = []
    
    # Prediction form
    st.subheader('Select Symptoms')
    
    for category, features in feature_categories.items():
        # Add a collapsible section for each feature category
        with st.beta_expander(category):
            for symptom in features:
                if st.checkbox(symptom):
                    if symptom not in session_id['selected_features']:
                        session_id['selected_features'].append(symptom)
                    else:
                        st.warning(f"{symptom} is already selected.")
                else:
                    if symptom in session_id['selected_features']:
                        session_id['selected_features'].remove(symptom)
    
    selected_features = session_id['selected_features']
    feature_vector = np.zeros(132)

    for symptom in selected_features:
        feature_index = all_features.index(symptom)
        feature_vector[feature_index] = 1

    # Make prediction
    prediction_proba = best_model.predict_proba([feature_vector])[0]
    prediction_class = best_model.classes_[np.argmax(prediction_proba)]
    max_proba = np.max(prediction_proba)

    # Output prediction result
    st.write(f"Predicted Disease Class: {prediction_class}")
    st.write(f"Probability: {max_proba:.2f}")

    # Recommendation for more symptoms if less than 4 symptoms selected
    if len(selected_features) < 4:
        st.write("We recommend selecting more symptoms for accurate results.")

    # Clear button
    if st.button('Clear Input'):
        # Clear selected features and reload page
        session_id['selected_features'] = []
        st.experimental_rerun()

# Run the app
if __name__ == '__main__':
    main()
