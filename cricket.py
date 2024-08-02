import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Define the teams and venues
teams = ["India", "Pakistan", "Afghanistan", "Bangladesh", "Australia", "Ireland",
         "New Zealand", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe",
         "Uganda", "Iceland"]

venues = ['The Rose Bowl', 'Eden Park', 'New Wanderers Stadium', 'County Ground',
          'Gahanga International Cricket Stadium', 'GB Oval', 'Sportpark Het Schootsveld',
          'Malahide', 'Amini Park', 'Gymkhana Club Ground', 'Sylhet International Cricket Stadium',
          'Providence Stadium', 'Scott Page Field', 'JSCA International Stadium Complex',
          'Queens Sports Club']

# Load the trained model and label encoders
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    st.error("Error: The file 'model.pkl' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Error: Failed to load the model. {e}")
    st.stop()

# Load label encoders
try:
    with open('label_encoder_venue.pkl', 'rb') as f:
        label_encoder_venue = pickle.load(f)
    with open('label_encoder_bat_first.pkl', 'rb') as f:
        label_encoder_bat_first = pickle.load(f)
    with open('label_encoder_bat_second.pkl', 'rb') as f:
        label_encoder_bat_second = pickle.load(f)
    with open('label_encoder_winner.pkl', 'rb') as f:
        label_encoder_winner = pickle.load(f)
    print("Label encoders loaded successfully.")
except FileNotFoundError:
    st.error("Error: One or more label encoder files were not found.")
    st.stop()
except Exception as e:
    st.error(f"Error: Failed to load label encoders. {e}")
    st.stop()

# Function to make a prediction
def predict_winner(venue, bat_first, bat_second):
    # Encode input data
    try:
        venue_encoded = label_encoder_venue.transform([venue])
        bat_first_encoded = label_encoder_bat_first.transform([bat_first])
        bat_second_encoded = label_encoder_bat_second.transform([bat_second])
    except ValueError as e:
        return f"Error: One or more input values are not recognized. {e}"

    # Prepare input DataFrame
    input_data = {
        'Venue': [venue_encoded[0]],
        'Bat First': [bat_first_encoded[0]],
        'Bat Second': [bat_second_encoded[0]]
    }
    input_df = pd.DataFrame(input_data)

    # Predict
    try:
        prediction = model.predict(input_df)
        winner = label_encoder_winner.inverse_transform(prediction)
        return winner[0]
    except Exception as e:
        return f"Error: Prediction failed. {e}"

# Streamlit application
st.title("Cricket Match Prediction")

# Create and place widgets
venue = st.selectbox("Select Venue:", venues)
bat_first = st.selectbox("Select Team Batting First:", teams)
bat_second = st.selectbox("Select Team Batting Second:", teams)

if st.button("Predict Winner"):
    winner = predict_winner(venue, bat_first, bat_second)
    st.write(f"**Predicted Winner:** {winner}")

