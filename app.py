import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Ice Cream Sales Predictor",
    page_icon="🍦"
)

CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}

URL = "https://api.open-meteo.com/v1/forecast"

# Load the model trained with both predictors.
model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🍦 Ice Cream Sales Predictor")

st.write(
    "Choose a city and date to estimate ice cream sales using "
    "temperature and precipitation probability."
)

st.caption(
    "This classroom model uses simulated data. Incorrect predictions "
    "could lead to preparing too much or too little stock."
)

city = st.selectbox("City", list(CITIES))

selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=6)
)

if st.button("Predict sales", type="primary"):
    latitude, longitude = CITIES[city]

    try:
        with st.spinner("Getting the weather forecast..."):
            response = requests.get(
                URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": (
                        "temperature_2m_max,"
                        "precipitation_probability_max"
                    ),
                    "temperature_unit": "fahrenheit",
                    "timezone": "auto",
                    "start_date": str(selected_date),
                    "end_date": str(selected_date),
                },
                timeout=30,
            )

            response.raise_for_status()
            daily = response.json()["daily"]

            temperature = daily["temperature_2m_max"][0]
            rain_probability = daily["precipitation_probability_max"][0]

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ):
        st.error(
            "Unable to retrieve the weather forecast. "
            "Please try again or choose another date."
        )
        st.stop()

    if temperature is None or rain_probability is None:
        st.error(
            "The forecast is incomplete. Please choose another date."
        )
        st.stop()

    # Match the predictor names and order used during training.
    features = pd.DataFrame({
        "temperature": [temperature],
        "rain_probability": [rain_probability],
    })

    try:
        prediction = max(0, float(model.predict(features)[0]))
    except ValueError:
        st.error(
            "The saved model does not match these inputs. "
            "Upload the updated model.pkl trained with temperature "
            "and rain_probability."
        )
        st.stop()

    st.subheader("Prediction results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast temperature",
        f"{temperature:.0f} °F"
    )

    col2.metric(
        "Precipitation probability",
        f"{rain_probability:.0f}%"
    )

    col3.metric(
        "Predicted sales",
        f"{prediction:.0f} cones"
    )

    if temperature < 50 or temperature > 100:
        st.warning(
            "This temperature is outside the model's training range "
            "of 50–100°F. Treat the estimate with caution."
        )

    if prediction > 300:
        st.info("Busy day expected: consider preparing more stock.")
    elif prediction > 200:
        st.info("A normal sales day is expected.")
    else:
        st.info("Slow day expected: consider preparing less stock.")

st.caption("Weather forecasts provided by Open-Meteo.")
st.markdown("[Weather data source](https://open-meteo.com/)")
