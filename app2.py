import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
# Load the historical data
histo = pd.read_csv('2nd_exp.csv')

# Define a function to predict the AQI
def predict_aqi(city, date, df):
    city_df = df[df['City']==city].copy()
    global model
    model = Prophet(
        seasonality_prior_scale=5.0,
    )
    model.fit(city_df)
    future = pd.DataFrame({'ds': pd.to_datetime([date])})
    forecast = model.predict(future)
    predicted = forecast.loc[0:'yhat']
    return predicted

# Create a Streamlit app
st.title('AQI Prediction App')

# Get the city from the user
city = st.text_input('Enter a city:')

# Get the date from the user
today = datetime.date.today()
d = today + datetime.timedelta(days=1)
date = st.date_input('Enter a date:', d)

# Make the prediction
pred = predict_aqi(city, date, histo)

# Display the prediction to the user
st.write('The predicted AQI for {} on {} is: {}'.format(city, date, abs(pred['yhat'][0])))
