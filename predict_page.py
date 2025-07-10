import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('C:/Users/SUNNY/Downloads/Software Developer Salary prediction/pre_saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    Countries = [
    'Australia',
    'Austria',
    'Belgium',
    'Brazil',
    'Canada',
    'Czech Republic',
    'Denmark',
    'Finland',
    'France',
    'Germany',
    'Greece',
    'Hungary',
    'India',
    'Israel',
    'Italy',
    'Mexico',
    'Netherlands',
    'New Zealand',
    'Norway',
    'Poland',
    'Portugal',
    'Russian Federation',
    'South Africa',
    'Spain',
    'Sweden',
    'Switzerland',
    'Turkey',
    'Ukraine',
    'United Kingdom of Great Britain and Northern Ireland',
    'United States of America'
    ]


    Education = [
        'Post grad', 
        'Master’s degree', 
        'Less than a Bachelors',  
        'Bachelor’s degree'
    ]

    country = st.selectbox("country", Countries)
    education = st.selectbox("Years of Experience", Education)

    experience = st.slider("YearsCodePro", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country, education, experience]])
        
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x = x.astype(float)

        y_pred = regressor.predict(x)
        
        st.subheader("Predicted Salary (USD): ${:,.02f}".format(y_pred[0]))

show_predict_page()
