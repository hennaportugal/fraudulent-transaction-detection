# Machine Learning Engineering Code Challenge

Fraud detection API that predicts whether or not a transaction is fraudulent

### [XGBoost](/https://xgboost.readthedocs.io/en/stable/python/index.html)
Machine learning algorithm used to build the model with the structured/tabular data
> *“When in doubt, use XGBoost” — Owen Zhang, Winner of Avito Context Ad Click Prediction competition on Kaggle*
##### See `model.py` for the model development (also data preparation and feature engineering)

### Deployed REST API
API takes a POST request to return a JSON object with a boolean value for its field __isFraud__\
The application is built with *Flask* and is deployed to *Heroku*
##### See `fraudDetection.py` for loading the classifier, data processing, and getting the prediction | See `app.py` for the API source code
