import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# from PIL import Image
# import base64

# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# bin_str = get_base64('website_background.jpg')
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{bin_str}");
#         background-size: cover;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Load the saved model
model = joblib.load('insurance_prediction.pkl')

# Define the function to get user input
# def get_user_input():

st.title('Insurance Outcome Prediction')

st.header('Please fill in the customer information')

st.text('The customer age is split into 4 categories:')
st.caption('0: the customer is between 17 and 25')
st.caption('1: the customer is between 26 and 35')
st.caption('2: the customer is between 36 and 45')
st.caption('3: the customer is between over 45')

age = st.number_input('Age Category:', value=0, min_value=0, max_value=3, step=1)
gender = st.number_input('Gender: 0 for female, 1 for male', value=0, min_value=0, max_value=1, step=1)
driving_experience = st.radio('Driving Experience (years)',['0-9y', '10-19y', '20-29y', '30y+'])
# exp = st.radio("driving exp",('0-9y', '10-19y', '20-29y', '30y+'))

education = st.selectbox('Education', ['high school', 'none', 'university'])
income = st.selectbox('Income', ['upper class', 'poverty', 'working class', 'middle class'])

st.text('kindly enter your credit score between 0 and 1000. the number will later be devided by 1000')
credit_score = st.slider('Credit Score:', min_value=0, max_value=1000, step=1)
vehicle_ownership = st.number_input('Vehicle Ownership: 1 for owned and 0 for borrowed', value=0, min_value=0, max_value=1, step=1)
vehicle_year = st.radio('Vehicle Year', ['after 2015', 'before 2015'])

married = st.number_input('Marital Status: 0 for single and 1 for Married', value=0, min_value=0, max_value=1, step=1)
children = st.number_input('Has Children: 0 for No and 1 for yes', value=0, min_value=0, max_value=1, step=1)
annual_mileage = st.slider('Annual Mileage', min_value=0,max_value= 80000, step=1000)
vehicle_type = st.radio('Vehicle Type', ['sedan', 'sports car'])
speeding_violations = st.slider('Speeding Violations', min_value=0, max_value = 25, step=1)
duis = st.number_input('DUIs', min_value=0, step=1)
past_accidents = st.slider('Past Accidents', min_value=0, max_value = 15,step=1)

# Create a dictionary with user input
user_data = {
    'age': age,
    'gender': gender,
    'driving_experience': driving_experience,
    'education': education,
    'income': income,
    'credit_score': credit_score/1000,
    'vehicle_ownership': vehicle_ownership,
    'vehicle_year': vehicle_year,
    'married': married,
    'children': children,
    'annual_mileage': annual_mileage,
    'vehicle_type': vehicle_type,
    'speeding_violations': speeding_violations,
    'duis': duis,
    'past_accidents': past_accidents
}


# Convert user input to a DataFrame
user_df = pd.DataFrame([user_data])

# Load the ColumnTransformer from the trained model
column_transformer = model.named_steps['transformer']

# Transform the user input data
transformed_data = column_transformer.transform(user_df)


# Display the prediction
if st.button("Predict"):
    # Make predictions
    prediction = model.named_steps['classifier'].predict(transformed_data)
    if prediction[0] == 0:
        st.write('The customer will most likely not file claims for their car')
    else:
        st.write('The customer will most likely file claims for their car')
        

st.caption(' ')

st.caption('Kindly Note that this Model is for Demostration only.')

        
