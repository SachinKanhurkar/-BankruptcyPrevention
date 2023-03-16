from flask import Flask, request, jsonify
import pickle
import csv
import pandas as pd
import mlflow

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('model_RandomForest.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
  
    json_ = request.get_json(force=True)
    # print(json_)

    query_df = pd.DataFrame(json_)  
    query1=query_df.values

    # query = pd.read_csv('datafile.csv')

    # query1=query.values

    print(query1)
    prediction = model.predict(query1)
 
    print(prediction)

    # Return the prediction as a JSON response
    if prediction ==0:
        output = {'prediction': str("Bankruptcy")}
    else:
        output = {'prediction': str("Non-bankruptcy")}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)