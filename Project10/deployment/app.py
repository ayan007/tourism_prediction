import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="ayanmalakar/tourism_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the probability of a customer purchasing a tourism package
""")

st.header("User Input")

# User input
# Numerical inputs
input_age               = st.number_input("Age", min_value=18, max_value=90, value=30)
input_num_persons       = st.number_input("Number of Persons Visiting", min_value=1, max_value=8, value=2)
input_num_children      = st.number_input("Number of Children Visiting", min_value=0, max_value=4, value=0)
input_num_trips         = st.number_input("Number of Trips", min_value=1, max_value=20, value=5)
input_monthly_income    = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=5000.0)
input_property_star     = st.number_input("Preferred Property Star", min_value=3, max_value=5, value=3)
input_num_followups     = st.number_input("Number of Followups", min_value=1, max_value=10, value=2)
input_duration_pitch    = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
input_pitch_satisfaction= st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
input_city_tier         = st.number_input("City Tier", min_value=1, max_value=3, value=1)

# Categorical inputs
input_gender          = st.selectbox("Gender", options=["Male", "Female"])
input_marital_status  = st.selectbox("Marital Status", options=["Married", "Unmarried", "Divorced"])
input_occupation      = st.selectbox("Occupation", options=["Salaried", "Free Lancer", "Small Business", "Large Business"])
input_designation     = st.selectbox("Designation", options=["AVP", "VP", "Executive", "Manager", "Senior Manager"])
input_passport        = st.selectbox("Passport", options=["Yes", "No"])
input_own_car         = st.selectbox("Own Car", options=["Yes", "No"])
input_type_of_contact = st.selectbox("Type of Contact", options=["Company Invited", "Self Enquiry"])
input_product_pitched = st.selectbox("Product Pitched", options=["Basic", "Standard", "King", "Deluxe", "Super Deluxe"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': input_age,
    'TypeofContact': input_type_of_contact,
    'CityTier': input_city_tier,
    'DurationOfPitch': input_duration_pitch,
    'Occupation': input_occupation,
    'Gender': input_gender,
    'NumberOfPersonVisiting': input_num_persons,
    'NumberOfFollowups': input_num_followups,
    'ProductPitched': input_product_pitched,
    'PreferredPropertyStar': input_property_star,
    'MaritalStatus': input_marital_status,
    'NumberOfTrips': input_num_trips,
    'Passport': input_passport,
    'PitchSatisfactionScore': input_pitch_satisfaction,
    'OwnCar': input_own_car,
    'NumberOfChildrenVisiting': input_num_children,
    'Designation': input_designation,
    'MonthlyIncome': input_monthly_income
}])

# Predict
if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = prediction * 100
    st.subheader("Prediction Result:")
    st.success(f"Predicted likelihood of purchasing the package: **{result:.2f}%**")
