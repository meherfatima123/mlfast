import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
import pickle
import warnings

warnings.filterwarnings("ignore")


def Classification(
    X, y, model="lr", scaler=None, cat=False, hyperparams=None, save_pkl=False
):
    """
    The Classification is one of the methods in mlfast library to build machine learning Models in an easy way.
    Parameters:

    X: The independent variables (features) of the data set.

    y: The dependent variable (target) of the data set.

    model: {lr, sgd, dt, rf, svm, knn, gb, ada, xgb, nb, mlp}. It takes at least one argument.
        `lr`: Logistic Regression
        `sgd`: Stochastic Gradient Descent
        `dt`: Decision tree classifier
        `rf`: Random forest classifier
        `svm`: Support vector machine
        `knn`: K Nearest Neighbours classifier
        `gb`: Gradient boosting classifier
        `ada`: ADA boosting classifier
        `xgb`: Extreme Gradient Boosting classifier
        `nb`: Naive Bayes
        `mlp`: Multi-layer Perceptron classifier

    scaler: {standard, robust} The name of the data scaler, by default, it will be None performed only on numerical data.
        `standard`: StandardScaler
        `robust`: RobustScaler

    cat: A boolean indicating [True or False] whether the data set has categorical variables that need to be one-hot encoded. By default, it is False.

    hyperparams: By default, it is None. The user needs to pass hyperparameters based on the model.

    save_pkl: A boolean value [True or False]. If it is True, it saves the model in a pickle file. By default, it is false.

    Returns:
    """

    import warnings

    warnings.filterwarnings("ignore")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52
    )

    # One-hot encoding
    if cat:
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            transformer = ColumnTransformer(
                transformers=[("ohe", OneHotEncoder(), categorical_cols)],
                remainder="passthrough",
            )
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)

    # Encoding the target variable if it is categorical
    if y_train.dtype == "O":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

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

        # Save the scaler as a pickle file
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
            clf = GridSearchCV(
                LogisticRegression(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = LogisticRegression()
    elif model == "sgd":
        if hyperparams:
            clf = GridSearchCV(
                SGDClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = SGDClassifier()
    elif model == "dt":
        if hyperparams:
            clf = GridSearchCV(
                DecisionTreeClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = DecisionTreeClassifier()
    elif model == "rf":
        if hyperparams:
            clf = GridSearchCV(
                RandomForestClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = RandomForestClassifier()
    elif model == "svm":
        if hyperparams:
            clf = GridSearchCV(
                SVC(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = SVC()
    elif model == "knn":
        if hyperparams:
            clf = GridSearchCV(
                KNeighborsClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = KNeighborsClassifier()
    elif model == "gb":
        if hyperparams:
            clf = GridSearchCV(
                GradientBoostingClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = GradientBoostingClassifier()
    elif model == "ada":
        if hyperparams:
            clf = GridSearchCV(
                AdaBoostClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = AdaBoostClassifier()
    elif model == "xgb":
        if hyperparams:
            clf = GridSearchCV(
                xgb.XGBClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.05, random_state=42
            )
    elif model == "nb":
        clf = GaussianNB()
    elif model == "mlp":
        if hyperparams:
            clf = GridSearchCV(
                MLPClassifier(),
                hyperparams,
                scoring="accuracy",
                cv=5,
                n_jobs=-1,
            )
        else:
            clf = MLPClassifier()
    else:
        return print("Invalid model name")

    # Fitting the Model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Printing Metrics
    print(
        f"Accuracy Score: {accuracy_score(y_test, y_pred)},\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}"
    )

    # Saving the Model
    if save_pkl:
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        filename = os.path.join(model_dir, f"{model}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(clf, file)
        print(f"Model saved successfully in {filename}")
