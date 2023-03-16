from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

#Create a Flask app instance and define the route for the home page:
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

#Define the route for the prediction page, where the user will input the values for the prediction:

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the input values from the user
    industrial_risk = float(request.form['industrial_risk'])
    management_risk = float(request.form['management_risk'])
    financial_flexibility = float(request.form['financial_flexibility'])
    credibility = float(request.form['credibility'])
    competitiveness = float(request.form['competitiveness'])
    operating_risk = float(request.form['operating_risk'])

    # Load the trained ML model
    model = joblib.load('model_RandomForest.pkl')

    # Make a prediction using the input values
    prediction = model.predict(np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]).reshape(1, -1))[0]
 
    print(prediction)

    # Return the prediction as a JSON response
    if prediction ==0:
        output = {'prediction': str("Bankruptcy")}
    else:
        output = {'prediction': str("Non-bankruptcy")}

    return  jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)