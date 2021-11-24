# IMPORTING PACKAGES
import pandas as pd # data processing
import numpy as np # working with arrays
import seaborn as sns

import joblib
import warnings

from termcolor import colored as cl # text customization
import itertools # advanced tools

from sklearn import preprocessing
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing import StandardScaler # data normalization
from category_encoders import OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm
from xgboost import XGBClassifier # XGBoost algorithm

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, classification_report # evaluation metric
from sklearn.metrics import recall_score, f1_score, make_scorer, cohen_kappa_score
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric
import inflection

warnings.filterwarnings('ignore')
seed = 42
np.random.seed(seed)

# IMPORTING DATA
data = pd.read_csv('transactions_train.csv')

# print(df.head())
# print(df.tail())

# # DATA COLUMNS
# cols_old = df.columns.tolist()

# snakecase = lambda x: inflection.underscore(x)
# cols_new = list(map(snakecase, cols_old))

# df.columns = cols_new
# print(df.columns)

# # DATA TYPE/STRUCTURE
# print('Number of Rows: {}'.format(df.shape[0]))
# print('Number of Cols: {}'.format(df.shape[1]))
# print(df.info())

# print(df.isna().mean()) # check NaN

data['isFraud'] = data['isFraud'].map({1: 'true', 0: 'false'}) # API is expected to return a JSON object with a boolean field isFraud

# # DESCRIPTION STAT
# num_attributes = df.select_dtypes(exclude='object')
# cat_attributes = df.select_dtypes(include='object')

# describe = num_attributes.describe().T

# describe['range'] = (num_attributes.max() - num_attributes.min()).tolist()
# describe['variation coefficient'] = (num_attributes.std() / num_attributes.mean()).tolist()
# describe['skew'] = num_attributes.skew().tolist()
# describe['kurtosis'] = num_attributes.kurtosis().tolist()

# # print(describe)

# print(cat_attributes.describe())

# FEATURE ENGINEERING
data1 = data.copy()
# step
data1['stepDays'] = data1['step'].apply(lambda i: i/24)
data1['stepWeeks'] = data1['step'].apply(lambda i: i/(24*7))

# difference between initial balance before the transaction and new balance after the transaction
data1['diff_new_old_balance'] = data1['newbalanceOrig'] - data1['oldbalanceOrig']

# difference between initial balance recipient before the transaction and new balance recipient after the transaction.
data1['diff_new_old_destiny'] = data1['newbalanceDest'] - data1['oldbalanceDest']

# name orig and name dest
data1['nameOrig'] = data1['nameOrig'].apply(lambda i: i[0])
data1['nameDest'] = data1['nameDest'].apply(lambda i: i[0])

# DATA PREPARATION
data2 = data1.copy()

X = data2.drop(columns=['isFraud', 'nameOrig', 'nameDest', 
                      'stepWeeks', 'stepDays'], axis=1)
y = data2['isFraud'].map({'true': 1, 'false': 0})

# spliting into temp and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=.2, stratify=y)

# spliting into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=.2, stratify=y_temp)

ohe = OneHotEncoder(cols=['type'], use_cat_names=True)

X_train = ohe.fit_transform(X_train)
X_valid = ohe.transform(X_valid)

X_temp = ohe.fit_transform(X_temp)
X_test = ohe.transform(X_test)

num_columns = ['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
               'diff_new_old_balance', 'diff_new_old_destiny']
mm = MinMaxScaler()
X_params = X_temp.copy()

X_train[num_columns] = mm.fit_transform(X_train[num_columns])
X_valid[num_columns] = mm.transform(X_valid[num_columns])

X_params[num_columns] = mm.fit_transform(X_temp[num_columns])
X_test[num_columns] = mm.transform(X_test[num_columns])

# FEATURE SELECTION
final_columns_selected = ['step', 'oldbalanceOrig', 
                          'newbalanceOrig', 'newbalanceDest', 
                          'diff_new_old_balance', 'diff_new_old_destiny', 
                          'type_TRANSFER']

# MACHINE LEARNING MODELING
X_train_cs = X_train[final_columns_selected]
X_valid_cs = X_valid[final_columns_selected]

X_temp_cs = X_temp[final_columns_selected]
X_test_cs = X_test[final_columns_selected]

X_params_cs = X_params[final_columns_selected]

# xgboost
xgb = XGBClassifier()
xgb.fit(X_train_cs, y_train)

y_pred = xgb.predict(X_valid_cs)

# score
def ml_scores(model_name, y_true, y_pred):
    
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return pd.DataFrame({'Balanced Accuracy': np.round(accuracy, 3), 
                         'Precision': np.round(precision, 3), 
                         'Recall': np.round(recall, 3),
                         'F1': np.round(f1, 3),
                         'Kappa': np.round(kappa, 3)}, 
                        index=[model_name])

xgb_results = ml_scores('XGBoost', y_valid, y_pred)
print(xgb_results)
print(classification_report(y_valid, y_pred))

# hyperparameter fine tuning
f1 = make_scorer(f1_score)
params = {
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eta': [0.3, 0.1, 0.01],
    'scale_pos_weight': [1, 774, 508, 99]
}

gs = GridSearchCV(XGBClassifier(), 
                  param_grid=params, 
                  scoring=f1, 
                  cv=StratifiedKFold(n_splits=5))
gs.fit(X_params_cs, y_temp)

best_params = gs.best_params_
print(best_params,gs.best_score_)

# RESULTS
# xgb_gs = XGBClassifier(
#     booster=best_params['booster'],
#     eta=best_params['eta'],
#     scale_pos_weight=best_params['scale_pos_weight']
# )
# xgb_gs.fit(X_train_cs, y_train)
# y_pred = xgb_gs.predict(X_valid_cs)

# # single results
# xgb_gs_results = ml_scores('XGBoost GS', y_valid, y_pred)
# print(xgb_gs_results)

final_model = XGBClassifier(
    booster=best_params['booster'],
    eta=best_params['eta'],
    scale_pos_weight=best_params['scale_pos_weight']
)

final_model.fit(X_params_cs, y_temp)

joblib.dump(final_model, 'model_cycle2.joblib')

mm = MinMaxScaler()
mm.fit(X_params_cs, y_temp)

joblib.dump(mm, 'minmaxscaler_cycle2.joblib')

joblib.dump(ohe, 'onehotencoder_cycle2.joblib')

# unseen data score
y_pred = final_model.predict(X_test_cs)
unseen_scores = ml_scores('unseen', y_test, y_pred)
print(unseen_scores)
