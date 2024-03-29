import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the best model
best_model = load('best_model.joblib')

# Define the list of top features
top_features = [
    "sinus_pressure", "internal_itching", "hip_joint_pain", "increased_appetite",
    "inflammatory_nails", "brittle_nails", "blood_in_sputum", "yellow_crust_ooze",
    "nodal_skin_eruptions", "unsteadiness", "muscle_weakness", "weight_loss",
    "fluid_overload.1", "prominent_veins_on_calf", "depression", "lack_of_concentration",
    "yellowing_of_eyes", "muscle_pain", "mild_fever", "neck_pain"
]

# Create a Streamlit app
def main():
    # Define selected_features outside the main function
    selected_features = []

    st.title('Disease Prediction System')

    # Display top features
    st.subheader('Top Features:')
    for i, feature in enumerate(top_features, start=1):
        st.write(f"{i}. {feature}")

    # Prediction form
    st.sidebar.title('Predict Disease')
    
    for i, feature in enumerate(top_features, start=1):
        selected = st.sidebar.checkbox(feature, key=i)
        if selected:
            selected_features.append(feature)

    if st.sidebar.button('Predict'):
        if len(selected_features) == 0:
            st.error('Please select at least one symptom.')
        else:
            # Create feature vector based on selected symptoms
            feature_vector = np.zeros(len(top_features))
            for symptom in selected_features:
                index = top_features.index(symptom)
                feature_vector[index] = 1

            # Predict disease using the model
            prediction = best_model.predict([feature_vector])
            st.success(f'Predicted Disease: {prediction[0]}')

if __name__ == '__main__':
    main()
