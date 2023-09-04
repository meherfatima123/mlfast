import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
import pickle
import warnings

warnings.filterwarnings("ignore")


def Regression(
    X, y, model="lr", scaler=None, cat=False, hyperparams=None, save_pkl=False
):
    """
    The Classification is one of the methods in the mlfast library to build machine learning Models in an easy way.
    Parameters:

        X: The independent variables (features) of the data set.

        y: The dependent variable (target) of the data set.

        model: {lr, sgd, dt, rf, svm, knn, gb, ada, xgb, nb, mlp}. It takes at least one argument.
            - `lr`: Logistic Regression
            - `sgd`: Stochastic Gradient Descent
            - `dt`: Decision tree classifier
            - `rf`: Random forest classifier
            - `svm`: Support vector machine
            - `knn`: K Nearest Neighbours classifier
            - `gb`: Gradient boosting classifier
            - `ada`: ADA boosting classifier
            - `xgb`: Extreme Gradient Boosting classifier
            - `nb`: Naive Bayes
            - `mlp`: Multi-layer Perceptron classifier

        scaler: {standard,robust} The name of the data scaler  by default it will be None performed only on numerical data.
            - `standard`: StandardScaler
            - `robust`: RobustScaler

        cat: A boolean indicating [True or False] whether the data set has categorical variables that need to be one-hot encoded. By default it is False.

        hyperparams: By default it is None. Users need to pass hyperparams based on model.
            - hyperparameters for `lr` model i.e Logistic Regression
                - Regularization parameter (C): It controls the inverse of the regularization strength. A smaller value of C increases the regularization strength, while a larger value decreases it.
                - Penalty (penalty): It determines the type of regularization used. Common options include L1 regularization (Lasso) and L2 regularization (Ridge).
                - Maximum number of iterations (max_iter): It specifies the maximum number of iterations taken for the solver to converge.
                - Solver: It determines the algorithm used for optimization. Popular choices include 'liblinear', 'lbfgs', 'sag', and 'newton-cg'.
                - Class weight (class_weight): It assigns weights to different classes to handle class imbalance issues.

            - `sgd`: Stochastic Gradient Descent
            - ...

        save_pkl: A boolean value [True or False]. If it is True, it saves the model in a pickle file. By default, it is false.

    Returns:"""

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52
    )

    # One-hot encoding
    if cat:
        categorical_cols = X.select_dtypes(include=["O"]).columns.tolist()
        if categorical_cols:
            transformer = ColumnTransformer(
                transformers=[("ohe", OneHotEncoder(), categorical_cols)],
                remainder="passthrough",
            )
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)

    # Scaling the data
    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler as a pickle file if specified
        if save_pkl:
            model_dir = "model"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            scaler_filename = os.path.join(model_dir, f"{model}_scaler.pkl")
            with open(scaler_filename, "wb") as scaler_file:
                pickle.dump(scaler, scaler_file)

    # Selecting the Model
    if model == "lr":
        if hyperparams:
            reg = GridSearchCV(
                LinearRegression(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = LinearRegression()
    elif model == "ridge":
        if hyperparams:
            reg = GridSearchCV(
                Ridge(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = Ridge()
    elif model == "lasso":
        if hyperparams:
            reg = GridSearchCV(
                Lasso(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = Lasso()
    elif model == "enet":
        if hyperparams:
            reg = GridSearchCV(
                ElasticNet(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = ElasticNet()
    elif model == "dt":
        if hyperparams:
            reg = GridSearchCV(
                DecisionTreeRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = DecisionTreeRegressor()
    elif model == "rf":
        if hyperparams:
            reg = GridSearchCV(
                RandomForestRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = RandomForestRegressor()
    elif model == "svm":
        if hyperparams:
            reg = GridSearchCV(
                SVR(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = SVR()
    elif model == "knn":
        if hyperparams:
            reg = GridSearchCV(
                KNeighborsRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = KNeighborsRegressor()
    elif model == "gb":
        if hyperparams:
            reg = GridSearchCV(
                GradientBoostingRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = GradientBoostingRegressor()
    elif model == "ada":
        if hyperparams:
            reg = GridSearchCV(
                AdaBoostRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = AdaBoostRegressor()
    elif model == "xgb":
        if hyperparams:
            reg = GridSearchCV(
                xgb.XGBRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.05, random_state=42
            )
    elif model == "et":
        if hyperparams:
            reg = GridSearchCV(
                ExtraTreesRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = ExtraTreesRegressor()
    elif model == "gnb":  # Gaussian Naive Bayes
        reg = GaussianNB()
    elif model == "mlp":  # Multi-Layer Perceptron Regressor
        if hyperparams:
            reg = GridSearchCV(
                MLPRegressor(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = MLPRegressor(max_iter=1000, random_state=42)
    elif model == "lasso_cv":  # Lasso CV
        reg = LassoCV()
    elif model == "ridge_cv":  # Ridge CV
        reg = RidgeCV()
    elif model == "enet_cv":  # ElasticNet CV
        reg = ElasticNetCV()
    elif model == "kridge":  # Kernel Ridge Regressor
        if hyperparams:
            reg = GridSearchCV(
                KernelRidge(),
                hyperparams,
                scoring="neg_mean_squared_error",
                cv=5,
                n_jobs=-1,
            )
        else:
            reg = KernelRidge()
    else:
        return print("Invalid model name")

    # Fitting the Model
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Printing Metrics
    print(
        f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)},\n"
        f"Mean Squared Error: {mean_squared_error(y_test, y_pred)},\n"
        f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))},\n"
        f"R2 Score: {r2_score(y_test, y_pred)}"
    )

    # Saving the Model if specified
    if save_pkl:
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_filename = os.path.join(model_dir, f"{model}_model.pkl")
        with open(model_filename, "wb") as model_file:
            pickle.dump(reg, model_file)

        print(f"Model saved successfully in {model_filename}")
