import numpy as np
import pickle
import pandas as pd
import json
#from flasgger import Swagger
import streamlit as st
import joblib

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

__model=None
# @app.route('/predict',methods=["Get"])


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./artifacts/banglore_home_price_model.joblib', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def predict_homeprice(location, sqft, bath, bhk):

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def main():
    st.title("Realestate rate predictor in Banglore")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Let's Estimate a Home Price in Banglore</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    location = st.selectbox("location", __locations)
    sqft = st.number_input("sqft", 1500)
    bhk = st.number_input("BHK", 1)
    bath = st.number_input("bathrooms", 1)
    result = ""
    if bath<=bhk+2:
        if st.button("Predict"):
            result = predict_homeprice(location, sqft, bath, bhk)
            st.success('The estimeted price is {}L'.format(result))
    else:
        st.error('Why too many bathrooms! Decrease the count and try again:)')
    if st.button("Created By"):
        st.text('Akhil pechetti')

if __name__ == '__main__':
    load_saved_artifacts()
    main()