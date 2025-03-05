import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load dataset
df = pd.read_csv('churn.csv')

# Streamlit UI
st.title("Let's make a churn prediction for the cab driver 🚕...")
st.subheader("Sample Dataframe for reference")
st.dataframe(df.iloc[:,1:])

# Load Models and Preprocessing Objects

with open('best_xgb_model.pkl', 'rb') as file:
    XGB_model = pickle.load(file)

with open('best_model_rf.pkl', 'rb') as file2:
    RF_model = pickle.load(file2)

with open("selected_features.pkl", 'rb') as feats:
    selected_features = pickle.load(feats)

with open('scaled_features.pkl', 'rb') as std_scaler:
    scaler = pickle.load(std_scaler)



st.divider()
st.subheader("Let's go...")

# Preprocessing function
def preprocess_input(user_input):
    """
    user_input: Dictionary containing feature values
    Returns: Scaled input data as a DataFrame
    """
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Keep only selected features
        input_df = input_df[selected_features]

        # Apply scaling
        scaled_input = scaler.transform(input_df)

        return scaled_input

    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# TAKING INPUT FROM USER  
user_input = {}

# First Feature: Binary (0 or 1)
user_input[selected_features[0]] = st.radio(f"{selected_features[0]}", [0, 1])

# Second Feature: Range (1 to 5)
user_input[selected_features[1]] = st.slider(f"{selected_features[1]}", min_value=1, max_value=5, step=1)

# Third Feature: Any whole number
user_input[selected_features[2]] = st.number_input(f"{selected_features[2]}", min_value=0, step=1, format="%d")

# Model Selection
select_model = st.radio("Select Model:", ["XGBoost", "Random Forest"])

# Predict on Button Click
if st.button("Predict"):
    processed_input = preprocess_input(user_input)

    if processed_input is not None:  # Ensure processing didn't fail
        if select_model == "XGBoost":
            prediction = XGB_model.predict(processed_input)
        else:
            prediction = RF_model.predict(processed_input)

        st.write("Here is my prediction...")
        st.write(prediction[0])

        if prediction[0] == 0:
            st.subheader("Driver will NOT churn ✅")
        else:
            st.subheader(" Driver will churn 🚨")


