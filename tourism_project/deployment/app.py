import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load trained model
model_path = hf_hub_download(
    repo_id="akashyadav2005/tourism_project_model",
    filename="best_tourism_project_model_v1.joblib"
)
model = joblib.load(model_path)

# Load threshold
try:
    threshold_path = hf_hub_download(
        repo_id="akashyadav2005/tourism_project_model",
        filename="threshold.txt"
    )
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
except:
    threshold = 0.45  # Default fallback

st.title("Tourism Purchase Prediction App")
st.write("""
This app predicts whether a customer will take the travel product package.
Enter customer details below:
""")

# User input
age = st.number_input("Age", 18, 100, 30)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0, 100, 15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 1)
number_of_followups = st.number_input("Number of Followups", 0, 10, 1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips", 0, 20, 1)
passport = st.selectbox("Has Passport?", [0, 1])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1])
children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 1000.0, 100000.0, 20000.0)

# Assemble input
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase"):
    # Get probability
    probability = model.predict_proba(input_data)[0][1]
    
    # Apply threshold
    prediction = 1 if probability >= threshold else 0
    
    # Show probability
    st.subheader("Prediction Result:")
    st.write(f"**Probability of taking package:** {probability:.2%}")
    st.write(f"**Threshold used:** {threshold:.2f}")
    
    if prediction == 1:
        st.success(" Customer WILL take the package!")
        st.balloons()
    else:
        st.error(" Customer will NOT take the package")
    
    # Show confidence
    if probability >= 0.7:
        st.info(" High confidence prediction")
    elif probability >= 0.4:
        st.warning(" Moderate confidence prediction")
    else:
        st.warning(" Low confidence prediction")
