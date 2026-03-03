import streamlit as st 
import joblib 
import pandas as pd

# Page config
st.set_page_config(page_title="COVID-19 Prediction App", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

model = load_model()

st.title("COVID-19 Prediction App")
st.write("Enter the details below to get prediction.")

# ================= INPUT SECTION =================

continent = st.selectbox("Continent", 
                         ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"])

country = st.text_input("Country")

population = st.number_input("Population", min_value=0)

date = st.date_input("Date")

new_cases = st.number_input("New Cases", min_value=0)
active_cases = st.number_input("Active Cases", min_value=0)
cases_per_million = st.number_input("Cases per Million", min_value=0.0)

total_cases = st.number_input("Total Cases", min_value=0)

new_deaths = st.number_input("New Deaths", min_value=0)
deaths_per_million = st.number_input("Deaths per Million", min_value=0.0)

total_deaths = st.number_input("Total Deaths", min_value=0)

tests_per_million = st.number_input("Tests per Million", min_value=0.0)
total_tests = st.number_input("Total Tests", min_value=0)

# ================= PREDICTION =================

if st.button("Predict"):

    input_df = pd.DataFrame({
        "continent": [continent],
        "country": [country],
        "population": [population],
        "date": [str(date)],
        "new_cases": [new_cases],
        "active_cases": [active_cases],
        "cases_per_million": [cases_per_million],
        "total_cases": [total_cases],
        "new_deaths": [new_deaths],
        "deaths_per_million": [deaths_per_million],
        "total_deaths": [total_deaths],
        "tests_per_million": [tests_per_million],
        "total_tests": [total_tests]
    })

    prediction = model.predict(input_df)

    st.success(f"Prediction Result: {prediction[0]}")