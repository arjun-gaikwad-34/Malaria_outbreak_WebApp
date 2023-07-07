import pickle

import numpy as np
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Utilities for api call to OpenWeather
api_key = '94be3acbad56a7e0ec88de00a773185a'
# Depickling the model
clf = pickle.load(open('knn_clf.pkl', 'rb'))

# Depickling the label_encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Depickling the scaler
with open('sst.pkl', 'rb') as file:
    sst = pickle.load(file)

@app.route("/") #decorators
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    input_data = list(request.form.values())
    city = request.form.get('city')
    positive = request.form.get('positive')
    pf = request.form.get('pf')
    rainfall = request.form.get('rainfall')
    print(f"city = {city}, positive = {positive}, pf = {pf}, rainfall = {rainfall}")
    weather_data = get_weather(city, api_key)
    if weather_data:
        temperature = weather_data["main"]["temp"]
        description = weather_data["weather"][0]["description"]
        print(f"The current temperature in {city} is {temperature} C.")
        print(f"Description: {description}")
    else:
        print("Failed to retrieve weather data.")

    minTemp = weather_data['main']['temp_min']
    maxTemp = weather_data['main']['temp_max']
    avgHumidity = weather_data['main']['humidity']


    predicted = clf.predict(sst.transform([[maxTemp, minTemp, avgHumidity, rainfall, positive, pf]]))
    print(f'predicted = {predicted}')
    print(f'Inverse transform = {label_encoder.inverse_transform(predicted)}')
    pred = label_encoder.inverse_transform(predicted)[0]
    return render_template('index.html', pred = pred)


def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            # Weather data retrieval successful
            temperature = data["main"]["temp"]
            weather_description = data["weather"][0]["description"]

            return data
                # "temperature": temperature,
                # "description": weather_description,


        else:
            # Weather data retrieval failed
            return None
    except requests.exceptions.RequestException as e:
        # Error occurred during API request
        print("Error:", e)
        return None

if __name__ == "__main__":
    app.run(debug = True)