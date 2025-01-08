import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load the saved models
random_forest_model = joblib.load('RandomForest_model.pkl')
decision_tree_model = joblib.load('DecisionTree_model.pkl')

# Define the Streamlit app
def main():
    # Title and Header Section
    st.title("✨Customer Churn Prediction App✨")
    st.subheader("Implementasi Artificial Intelligence untuk Memprediksi Risiko Churn Nasabah Bank")
    st.markdown("Bertujuan untuk menganalisis pola perilaku nasabah dan mengidentifikasi tanda-tanda yang menunjukkan kemungkinan mereka akan menghentikan penggunaan layanan bank. Dengan menggunakan algoritma machine learning dan analisis data, bank dapat mengantisipasi kehilangan nasabah dan mengambil langkah-langkah proaktif untuk meningkatkan retensi nasabah.")
    
    # Adding a logo or header image
    st.image("C:/Project AI/image.png", use_column_width=True)

    # Allow the user to select the algorithm
    st.sidebar.header("Model Selection")
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Decision Tree"])

    # Input Section
    st.header("📝 Input Customer Data")
    st.markdown("Please fill out the following details about the customer:")
    
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Balance (USD)", min_value=0.0, value=10000.0)
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary (USD)", min_value=0.0, value=50000.0)

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

    # Prediction Section
    st.header("🔮 Prediction Results")
    if st.button("Predict Churn"):
        # Select model based on user choice
        if algorithm == "Random Forest":
            model = random_forest_model
        else:
            model = decision_tree_model

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error(f"❌ The customer is *likely to churn*.")
        else:
            st.success(f"✅ The customer is *likely to retained*.")

    # Footer Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.image("C:/Project AI/image 2.png", use_column_width=True)
    st.sidebar.info("© 2025 - Artificial Intelligence | Group 3 2023C")

if __name__ == "__main__":
    main()