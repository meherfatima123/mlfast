from mlfast.Hyperparameter import Regression
from mlfast.Hyperparameter import Classification


import pandas as pd

df = pd.read_csv("ParisHousing.csv")

X = df.iloc[:,:-1]


y = df.iloc[:,-1]



# Random Forest Regression

try:

    hyperparams_rf = {
        "n_estimators": [5, 10, 20],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    Regression(X, y, model="rf", scaler="standard",  hyperparams=hyperparams_rf, save_pkl=False)
except Exception as e:
    TypeError