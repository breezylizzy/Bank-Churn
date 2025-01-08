import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('C:/Users/eliza/OneDrive/Documents/PYTHON PROJECT/sem 3/Project AI/RandomForest_model.pkl')

# Define the Streamlit app
def main():
    st.title("Churn Prediction App")
    st.write("This app predicts customer churn based on input features.")

    # Input fields for user data
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=10000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    # Encode categorical features
    geography_mapping = {"France": 0, "Germany": 1, "Spain": 2}
    gender_mapping = {"Male": 1, "Female": 0}
    has_cr_card_mapping = {"Yes": 1, "No": 0}
    is_active_member_mapping = {"Yes": 1, "No": 0}

    geography_encoded = geography_mapping[geography]
    gender_encoded = gender_mapping[gender]
    has_cr_card_encoded = has_cr_card_mapping[has_cr_card]
    is_active_member_encoded = is_active_member_mapping[is_active_member]

    # Create input array
    input_data = np.array([
        credit_score,
        geography_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card_encoded,
        is_active_member_encoded,
        estimated_salary
    ]).reshape(1, -1)

    # Predict button
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error(f"The customer is likely to churn. (Probability: {probability[0][1]:.2f})")
        else:
            st.success(f"The customer is not likely to churn. (Probability: {probability[0][0]:.2f})")

if __name__ == "__main__":
    main()
