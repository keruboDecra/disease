import streamlit as st
import numpy as np
from joblib import load

# Suppress warning about invalid feature names
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the best model
best_model = load('best_model.joblib')

# Define feature categories
feature_categories = {
    "Symptoms": [
        'loss_of_smell', 'internal_itching', 'hip_joint_pain', 'increased_appetite', 'malaise', 
        'inflammatory_nails', 'enlarged_thyroid', 'blood_in_sputum', 'yellow_crust_ooze', 
        'nodal_skin_eruptions', 'unsteadiness', 'irritability', 'weight_loss', 
        'prominent_veins_on_calf', 'fluid_overload.1', 'depression', 'lack_of_concentration', 
        'muscle_pain', 'mild_fever', 'neck_pain', 'altered_sensorium', 'back_pain', 
        'slurred_speech', 'movement_stiffness', 'shivering', 'belly_pain', 
        'continuous_feel_of_urine', 'itching', 'rusty_sputum', 'receiving_unsterile_injections', 
        'patches_in_throat', 'dark_urine', 'nausea', 'family_history', 'stomach_pain', 
        'loss_of_appetite', 'stomach_bleeding', 'pus_filled_pimples', 'bloody_stool', 
        'sunken_eyes', 'breathlessness', 'dischromic_patches', 'abdominal_pain', 
        'continuous_sneezing', 'knee_pain', 'dehydration', 'blackheads', 'sweating', 
        'burning_micturition', 'joint_pain', 'weakness_of_one_body_side', 
        'red_spots_over_body', 'extra_marital_contacts', 'chest_pain', 'spinning_movements', 
        'diarrhoea', 'bladder_discomfort', 'high_fever', 'red_sore_around_nose', 
        'yellowing_of_eyes', 'yellowish_skin', 'watering_from_eyes', 'indigestion', 
        'headache', 'skin_peeling', 'constipation', 'scurring', 'pain_behind_the_eyes', 
        'silver_like_dusting', 'cough', 'vomiting', 'skin_rash', 'blister', 
        'small_dents_in_nails', 'chills', 'acidity', 'ulcers_on_tongue', 
        'muscle_wasting', 'spotting_urination', 'irregular_sugar_level', 'fatigue', 
        'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'restlessness', 
        'lethargy', 'yellow_urine', 'throat_irritation', 'acute_liver_failure', 
        'loss_of_balance', 'excessive_hunger', 'drying_and_tingling_lips', 'palpitations', 
        'history_of_alcohol_consumption', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
        'distention_of_abdomen', 'coma', 'brittle_nails', 'foul_smell_of_urine', 
        'passage_of_gases', 'receiving_blood_transfusion', 'toxic_look_typhos', 
        'visual_disturbances', 'mucoid_sputum', 'polyuria', 'abnormal_menstruation', 
        'swollen_extremeties', 'painful_walking', 'fluid_overload', 'weakness_in_limbs', 
        'swelling_of_stomach', 'swelled_lymph_nodes', 'blurred_and_distorted_vision', 
        'phlegm', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
        'fast_heart_rate', 'puffy_face_and_eyes', 'pain_during_bowel_movements', 
        'pain_in_anal_region', 'irritation_in_anus', 'dizziness', 'bruising', 
        'obesity', 'swollen_legs', 'swollen_blood_vessels', 'cramps'
    ],
    # Add more categories if needed
}

# Create a Streamlit app
def main(feature_categories):
    # Define selected_features outside the main function
    selected_features = []

    st.title('Disease Prediction System')

    # Prediction form
    st.sidebar.title('Predict Disease')
    
    for category, features in feature_categories.items():
        st.sidebar.subheader(category)
        for i, feature in enumerate(features, start=1):
            selected = st.sidebar.checkbox(feature, key=f"{category}-{i}")
            if selected:
                selected_features.append(feature)

    if st.sidebar.button('Predict'):
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
            st.success(f'Predicted Disease: {prediction[0]}')

if __name__ == '__main__':
    main(feature_categories)
