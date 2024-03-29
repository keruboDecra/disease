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
        'malaise', 'irritability', 'family_history', 'scurring', 'blackheads', 'sweating', 
        'burning_micturition', 'red_spots_over_body', 'extra_marital_contacts', 'phlegm', 
        'muscle_wasting', 'weight_gain'
    ]
}

# Create a Streamlit app
def main(feature_categories):
    # Define selected_features outside the main function
    selected_features = []

    st.title('Disease Prediction System')
    st.write('We are sad that you or your loved ones are unwell, thank you for choosing us toward your healing journey.')

    # Prediction form
    st.subheader('Select Symptoms')
    
    for category, features in feature_categories.items():
        # Add a collapsible section for each category
        with st.expander(category):
            for i, feature in enumerate(features, start=1):
                selected = st.checkbox(feature, key=f"{category}-{i}")
                if selected:
                    selected_features.append(feature)
    
    # Add slider to adjust threshold
    threshold = st.slider("Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.1, format="%.1f")
    
    if st.button('Predict'):
        # Check if at least one symptom is selected
        if len(selected_features) == 0:
            st.warning("Please select at least one symptom.")
        else:
            # Create feature vector based on selected symptoms
            feature_vector = np.zeros(132)  # Ensure feature vector length matches the model's input size
            for symptom in selected_features:
                for category_features in feature_categories.values():
                    if symptom in category_features:
                        index = category_features.index(symptom)
                        feature_vector[index] = 1

            # Predict disease using the model
            prediction_probabilities = best_model.predict_proba([feature_vector])[0]
            
            # Set a threshold probability for prediction
            predicted_diseases = [disease for disease, prob in zip(best_model.classes_, prediction_probabilities) if prob > threshold]
            
            # Output prediction even if less than 4 symptoms are selected
            st.success(f'Predicted Diseases (above {threshold * 100}% probability): {predicted_diseases}')
            
            if 1 <= len(selected_features) < 4:
                st.warning("Are you experiencing any other symptom? We recommend that you add it so we don't miss the accurate prediction. More than 4 symptoms tend to improve accuracy")
                st.write("Based on the selected symptoms, we recommend consulting a healthcare professional for further evaluation and diagnosis. Get well soon!")
    
    # Add button to clear input selections
    if st.button("Clear Input"):
        selected_features.clear()

# Add logo to sidebar
st.sidebar.image("pic.png", use_column_width=True)
# Add sidebar with optional buttons and logo
st.sidebar.title("Options")
if st.sidebar.button("Get a Health Professional Around Me"):
    st.sidebar.write("Please visit [this page](https://rw.britam.com/help/other-services/locate-a-branch) to locate a branch.")
if st.sidebar.button("Translate Symptoms to English"):
    st.sidebar.write("Please visit [this page](https://translate.google.com/?sl=rw&tl=en&text=umva&op=translate) to translate your symptoms.")



if __name__ == '__main__':
    if len(feature_categories) == 0:
        st.warning("Please select at least one symptom.")
    else:
        main(feature_categories)
