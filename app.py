import streamlit as st
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
        'increased_appetite', 'malaise', 'enlarged_thyroid', 'unsteadiness', 'belly_pain', 
        'continuous_feel_of_urine', 'receiving_unsterile_injections', 'dark_urine', 'family_history', 
        'stomach_bleeding', 'pus_filled_pimples', 'sunken_eyes', 'dischromic _patches', 'continuous_sneezing', 
        'knee_pain', 'dehydration', 'blackheads', 'burning_micturition', 'red_spots_over_body', 
        'extra_marital_contacts', 'spinning_movements', 'bladder_discomfort', 'red_sore_around_nose', 
        'yellowish_skin', 'indigestion', 'headache', 'constipation', 'scurring', 'pain_behind_the_eyes', 
        'silver_like_dusting', 'vomiting', 'small_dents_in_nails', 'chills', 'acidity', 'ulcers_on_tongue', 
        'muscle_wasting', 'spotting_ urination', 'fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 
        'yellow_urine', 'throat_irritation', 'acute_liver_failure', 'loss_of_balance', 'excessive_hunger', 
        'drying_and_tingling_lips', 'palpitations', 'history_of_alcohol_consumption', 'stiff_neck', 
        'distention_of_abdomen', 'coma', 'brittle_nails', 'foul_smell_of urine', 'passage_of_gases', 
        'receiving_blood_transfusion', 'toxic_look_(typhos)', 'mucoid_sputum', 'polyuria', 'abnormal_menstruation', 
        'swollen_extremeties', 'painful_walking', 'fluid_overload', 'weakness_in_limbs', 'swelling_of_stomach', 
        'swelled_lymph_nodes', 'blurred_and_distorted_vision', 'phlegm', 'redness_of_eyes', 'runny_nose', 'puffy_face_and_eyes', 
        'pain_during_bowel_movements', 'pain_in_anal_region', 'irritation_in_anus', 'dizziness', 'bruising', 'obesity', 
        'swollen_legs', 'swollen_blood_vessels', 'cramps'
    ]
}

# Create a Streamlit app
def main(feature_categories):
    # Define selected_features outside the main function
    selected_features = []

    st.title('Disease Prediction System')

    # Prediction form
    st.subheader('Select Symptoms')
    
    for category, features in feature_categories.items():
        # Add a collapsible section for each category
        with st.expander(category):
            for i, feature in enumerate(features, start=1):
                selected = st.checkbox(feature, key=f"{category}-{i}")
                if selected:
                    selected_features.append(feature)

    if st.button('Predict'):
        if len(selected_features) == 0:
            st.error('Please select at least one symptom.')
        else:
            # Create feature vector based on selected symptoms
            feature_vector = np.zeros(sum(len(features) for features in feature_categories.values()))
            for symptom in selected_features:
                for category_features in feature_categories.values():
                    if symptom in category_features:
                        index = category_features.index(symptom)
                        feature_vector[index] = 1

            # Predict disease using the model
            prediction = best_model.predict([feature_vector])
            st.success(f'Predicted Disease: {
