# 🍦 Ice Cream Sales Predictor

This Streamlit app estimates daily ice cream sales using the weather forecast for a selected city and date.

## Predictors

- **Temperature (°F):** Higher temperatures are assumed to increase sales.
- **Precipitation probability (%):** A higher chance of precipitation is assumed to decrease sales.

The original model used temperature only. For this assignment, I added precipitation probability, retrained the linear regression model in Google Colab, saved it as `model.pkl`, and updated the app to use both predictors.

## How to Use

1. Select a city and date.
2. Click **Predict sales**.
3. View the forecast temperature, precipitation probability, and predicted number of cones sold.

## Files

- `app.py`: Streamlit app that retrieves weather forecasts and predicts sales.
- `model.pkl`: Saved model trained with both predictors.
- `requirements.txt`: Packages needed to run the app.
- `Ice_Cream_Model_Updated.ipynb`: Notebook used to generate sample data, train, evaluate, and save the model.

## Data and Limitations

The model uses **simulated training data**, following the classroom example. The relationships between weather and sales are assumptions, not findings from real sales records.


Predictions are for educational purposes. Incorrect estimates could lead to preparing too much or too little stock. Temperatures outside the training range of **50–100°F** may produce less reliable estimates.

