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
    # Add other categories...
}

# Create a Streamlit app
def main(feature_categories):
    # Define selected_features outside the main function
    selected_features = []

    st.title('Welcome to the Disease Prediction System')

    # Loading page with welcoming message
    with st.spinner('We are sad to know you and your loved ones are unwell.'):
        time.sleep(3)  # Simulate loading for 3 seconds

    # Prediction form
    st.subheader('Please select symptoms to predict the disease')
    
    for category, features in feature_categories.items():
        # Add a collapsible section for each category
        with st.expander(category):
            for i, feature in enumerate(features, start=1):
                selected = st.checkbox(feature, key=f"{category}-{i}")
                if selected:
                    selected_features.append(feature)

    if st.button('Predict'):
        if len(selected_features) < 4:
            st.warning('For accurate prediction, please select at least 4 symptoms.')
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
            threshold = 0.3
            predicted_diseases = [disease for disease, prob in zip(best_model.classes_, prediction_probabilities) if prob > threshold]
            
            if not predicted_diseases:
                st.warning('No disease prediction above the threshold probability.')
            else:
                st.success(f'Predicted Diseases (above {threshold * 100}% probability): {predicted_diseases}')

if __name__ == '__main__':
    main(feature_categories)
