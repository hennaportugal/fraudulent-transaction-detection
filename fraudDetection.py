import joblib
import inflection
import pandas as pd
from flask import jsonify

class fraudDetection:
    
    def __init__(self):
        self.minmaxscaler = joblib.load('minmaxscaler_cycle2.joblib')
        self.onehotencoder = joblib.load('onehotencoder_cycle2.joblib')
        
    def data_cleaning(self, data1):
        cols_old = data1.columns.tolist()
        
        camelCase = lambda i: inflection.camelize(i, False)
        cols_new = list(map(camelCase,cols_old))
        
        data1.columns = cols_new
        
        return data1
    
    def feature_engineering(self, data2):
        # step
        data2['stepDays'] = data2['step'].apply(lambda i: i/24)
        data2['stepWeeks'] = data2['step'].apply(lambda i: i/(24*7))

        # difference between initial balance before the transaction and new balance after the transaction
        data2['diff_new_old_balance'] = data2['newbalanceOrig'] - data2['oldbalanceOrig']

        # difference between initial balance recipient before the transaction and new balance recipient after the transaction.
        data2['diff_new_old_destiny'] = data2['newbalanceDest'] - data2['oldbalanceDest']

        # name orig and name dest
        data2['nameOrig'] = data2['nameOrig'].apply(lambda i: i[0])
        data2['nameDest'] = data2['nameDest'].apply(lambda i: i[0])
        
        return data2.drop(columns=['nameOrig', 'nameDest', 
                      'stepWeeks', 'stepDays'], axis=1)
    
    def data_preparation(self, data3):
        # OneHotEncoder
        data3 = self.onehotencoder.transform(data3)

        # Rescaling 
        num_columns = ['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
               'diff_new_old_balance', 'diff_new_old_destiny']
        data3[num_columns] = self.minmaxscaler.transform(data3[num_columns])
        
        # selected columns
        final_columns_selected = ['step', 'oldbalanceOrig', 
                          'newbalanceOrig', 'newbalanceDest', 
                          'diff_new_old_balance', 'diff_new_old_destiny', 
                          'type_TRANSFER']
        return data3[final_columns_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        if pred==0:
            original_data.is_fraud = bool(0)
        elif pred==1:
            original_data.is_fraud = bool(1)
        
        return original_data.is_fraud