#Install Libraries
from flask import Flask, request, Response
import joblib
import pandas as pd
import numpy as np
import sys
from fraudDetection import fraudDetection

# loading model
model = joblib.load('model_cycle2.joblib')

# initialize API
app = Flask(__name__)

@app.route('/is-fraud', methods=['POST'])
def churn_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Rossmann class
        pipeline = fraudDetection()
        
        # data cleaning
        data1 = pipeline.data_cleaning(test_raw)
        print(test_raw)
        
        # feature engineering
        data2 = pipeline.feature_engineering(data1)
        print(data2)
        
        # data preparation
        data3 = pipeline.data_preparation(data2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, data3)
        
        return {"isFraud" : df_response}
        
        
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0',debug=True) 