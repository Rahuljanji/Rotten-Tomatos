# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the model and dataset (modify the paths as needed)
data = pd.read_excel(r'Rotten_Tomatoes_Movies3.xls')  # Replace with the actual dataset path


# Pre-trained model loading (ensure the model is saved as 'model.pkl')
import pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Audience Rating Prediction")
    st.write("Predict audience ratings for movies based on features.")

    # Input fields for features
    runtime_in_minutes = st.number_input("Runtime in Minutes", min_value=0, step=1)
    tomatometer_rating = st.number_input("Tomatometer Rating", min_value=0, max_value=100, step=1)
    tomatometer_count = st.number_input("Tomatometer Count", min_value=0, step=1)
    rating = st.selectbox("Movie Rating", data['rating'].dropna().unique())
    genre = st.selectbox("Genre", data['genre'].dropna().unique())
    directors = st.selectbox("Directors", data['directors'].dropna().unique())
    studio_name = st.selectbox("Studio Name", data['studio_name'].dropna().unique())
    tomatometer_status = st.selectbox("Tomatometer Status", data['tomatometer_status'].dropna().unique())

    # Predict button
    if st.button("Predict Audience Rating"):
        # Prepare the input data
        input_data = pd.DataFrame({
            'runtime_in_minutes': [runtime_in_minutes],
            'tomatometer_rating': [tomatometer_rating],
            'tomatometer_count': [tomatometer_count],
            'rating': [rating],
            'genre': [genre],
            'directors': [directors],
            'studio_name': [studio_name],
            'tomatometer_status': [tomatometer_status]
        })

        # Predict using the model
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Audience Rating: {prediction:.2f}")

if __name__ == '__main__':
    main()
