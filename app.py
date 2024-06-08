from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    designation = request.form['DESIGNATION']
    ratings = float(request.form['RATINGS'])
    past_exp = float(request.form['PAST EXP'])
    days_in_company = float(request.form['DAYS IN COMPANY'])

    # Create a DataFrame for the input
    input_df = pd.DataFrame([[ratings, past_exp, days_in_company, designation]],
                            columns=['RATINGS', 'PAST EXP', 'DAYS IN COMPANY', 'DESIGNATION'])

    # Make prediction
    prediction = model.predict(input_df)

    # Render the prediction result
    return render_template('index.html', text=f'Predicted Salary: {prediction[0]:.2f} $')

if __name__ == "__main__":
    app.run(debug=True, port=5002)  # Change the port number if needed
