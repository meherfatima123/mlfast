# mlfast

This Python machine learning package is built on top of scikit-learn and provides a simple API for regression and classification modeling. The main function in this package is called Regression() and Classification() which takes in the following arguments:

`X`: The independent variables (features) of the data set

`y`: The dependent variable (target) of the data set

`model`: The name of the regression and classification algorithm to be used (e.g. `lr` for Linear Regression, or `rf` for Random Forest Classifier, etc.)

`scaler`: The name of the data scaler to be used (e.g. "standard" for StandardScaler, "robust" for RobustScaler, etc.)

`cat`: A boolean indicating [`True` or `False`] whether the data set has categorical variables that need to be one-hot encoded



- PYPI link for this package - [mlfast](https://pypi.org/project/mlfast/)


## Getting Started

### Installations

!!! note "Installation steps"
    First let's do an easy pip installation of the library by running the following command -
    ```python
    pip install mlfast
    ```

## Usage

### Regression Algorithms

!!! note "For Regression Modeling"
    Import Regression Model -
    ```python
    from mlfast import Regression
    ```





**Linear Regression**  -> 'lr' 
```python

Regression(X, y, model = 'lr')

```


**Ridge Regression**  -> 'ridge ' 
```python

Regression(X, y, model = 'ridge', scaler =  'standard')

```

**Lasso Regression**  -> 'lasso' 
```python

Regression(X, y, model = 'lasso', scaler =  'robust')

```

**ElasticNet**  -> 'enet' 
```python

Regression(X, y, model = 'enet', cat=True)

```


**Random Forest Regressor**  -> 'rf' 
```python

Regression(X, y, model = 'rf',scaler = 'standard', cat=True)

```



**Decision Tree Regressor**  -> 'dt' 
```python

Regression(X, y, model = 'dt')

```


**Support Vector Machine Regression**  -> 'svm ' 
```python

Regression(X, y, model = 'svm', scaler =  'standard')

```

**KNeighbors Regressor**  -> 'knn' 
```python

Regression(X, y, model = 'knn', scaler =  'robust')

```

**Gradient Boosting Regressor**  -> 'gb' 
```python

Regression(X, y, model = 'gb', cat=True)

```


**AdaBoost Regressor**  -> 'ada' 
```python

Regression(X, y, model = 'ada',scaler = 'standard', cat=True)

```


**XGBoost Regressor**  -> 'xgb' 
```python

Regression(X, y, model = 'xgb',scaler = 'standard', cat=True)

```





### Classification Algorithms


!!! note "For Classification Modeling"
    Import Classification Model -
    ```python
    from mlfast import Classification
    ```





**Logistic Regression**  -> 'lr' 
```python

Classification(X, y, model = 'lr')

```


**Random Forest Classifier**  -> 'rf' 
```python

Classification(X, y, model = 'rf',scaler = 'standard', cat=True)

```



**Decision Tree Classifier**  -> 'dt' 
```python

Classification(X, y, model = 'dt')

```


**Support Vector Machine Classifier**  -> 'svm ' 
```python

Classification(X, y, model = 'svm', scaler =  'standard')

```

**KNeighbors Classifier**  -> 'knn' 
```python

Classification(X, y, model = 'knn', scaler =  'robust')

```

**Gradient Boosting Classifier**  -> 'gb' 
```python

Classification(X, y, model = 'gb', cat=True)

```


**AdaBoost Classifier**  -> 'ada' 
```python

Classification(X, y, model = 'ada',scaler = 'standard', cat=True)

```


**XGBoost Classifier**  -> 'xgb'

```python

Classification(X, y, model = 'xgb',scaler = 'standard', cat=True)
```






### Regression hyperparameter


!!! note "For Regression Hyperparameter Modeling"
    Import Regression Hyperparameter Model -
    ```python
    from mlfast.Hyperparameter.Regression import Regression
    ```










**Ridge regression**


```python
hyperparams_ridge = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
    "fit_intercept": [True, False],
    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
}

Regression(X, y, model="ridge", scaler="standard", hyperparams=hyperparams_ridge)
```




**Lasso Regression**

```python
hyperparams_lasso = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
}
Regression(X, y, model="lasso", scaler="standard", hyperparams=hyperparams_lasso, save_pkl=False)
```


**ElasticNet Regression**

```python
hyperparams_enet = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.25, 0.5, 0.75],
}
Regression(X, y, model="enet", scaler="standard", hyperparams=hyperparams_enet, save_pkl=False)
```


**Decision Tree Regression**

```python
hyperparams_dt = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
Regression(X, y, model="dt", scaler="standard",  hyperparams=hyperparams_dt, save_pkl=False)
```


**Random Forest Regression**

```python
hyperparams_rf = {
    "n_estimators": [5, 10, 20],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
Regression(X, y, model="rf", scaler="standard",  hyperparams=hyperparams_rf, save_pkl=False)
```


**Support Vector Regression (SVR)**

```python
hyperparams_svm = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["linear", "rbf", "poly"],
}
Regression(X, y, model="svm", scaler="standard",  hyperparams=hyperparams_svm, save_pkl=False)
```


**K-Nearest Neighbors Regression (KNN)**

```python
hyperparams_knn = {
    "n_neighbors": [2, 3, 5],
    "weights": ["uniform", "distance"],
}
Regression(X, y, model="knn", scaler="standard",  hyperparams=hyperparams_knn, save_pkl=False)
```


**Gradient Boosting Regression**

```python
hyperparams_gb = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Regression(X, y, model="gb", scaler="standard", hyperparams=hyperparams_gb, save_pkl=False)
```


**AdaBoost Regression**

```python
hyperparams_ada = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Regression(X, y, model="ada", scaler="standard",  hyperparams=hyperparams_ada, save_pkl=False)
```

**XGBoost Regression**

```python
hyperparams_xgb = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Regression(X, y, model="xgb", scaler="standard",  hyperparams=hyperparams_xgb, save_pkl=False)
```




















### Classification Hyperparameter 



!!! note "For Classification Modeling"
    Import Classification Model -
    ```python
    from mlfast.Hyperparameter.Classification import Classification
    ```





**Logistic Regression**

```python
hyperparams_lr = {
    "C": [0.1, 1.0],
}
Classification(X, y, model="lr", scaler="robust", hyperparams=hyperparams_lr, save_pkl=True)
```

**Decision Tree Classifier**

```python
hyperparams_dt = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
Classification(X, y, model="dt", scaler="standard", hyperparams=hyperparams_dt)
```


**Random Forest Classifier**

```python
rf_hyperparams = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

Classification(X, y, model="rf", scaler="standard", cat=True, hyperparams=rf_hyperparams, save_pkl=False)
```


**Support Vector Classifier (SVC)**

```python
hyperparams_svm = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["linear", "rbf", "poly"],
}
Classification(X, y, model="svm", scaler="standard", hyperparams=hyperparams_svm)
```


**K-Nearest Neighbors Classifier (KNN)**

```python
hyperparams_knn = {
    "n_neighbors": [3, 5, 10],
    "weights": ["uniform", "distance"],
}
Classification(X, y, model="knn", scaler="standard", hyperparams=hyperparams_knn)
```



**Gradient Boosting Classifier**

```python
hyperparams_gb = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Classification(X, y, model="gb", scaler="standard", hyperparams=hyperparams_gb)
```



**AdaBoost Classifier**

```python
hyperparams_ada = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Classification(X, y, model="ada", scaler="standard", cat=True, hyperparams=hyperparams_ada)
```


**XGBoost Classifier**

```python
hyperparams_xgb = {
    "n_estimators": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.2],
}
Classification(X, y, model="xgb", scaler="standard", hyperparams=hyperparams_xgb)
```















### Text Preprocessing


The provided code snippet applies Text **Preprocessing** to a series or column of text data. It allows for flexible control over different preprocessing steps through boolean flags. Here is a summary of the options:

- `stem`: Determines whether stemming should be performed. Set to `True` to enable stemming, or `False` to disable it.
- `lemmatize`: Controls lemmatization. Set to `True` to enable lemmatization, or `False` to disable it.
- `remove_html`: Specifies whether HTML tags should be removed. Use `True` to remove HTML tags, or `False` to keep them.
- `remove_emoji`: Determines whether emojis should be removed from the text. Set to `True` to remove emojis, or `False` to retain them.
- `remove_special_chars`: Controls the removal of special characters. Use `True` to remove special characters, or `False` to keep them.
- `remove_extra_spaces`: Specifies whether extra spaces should be removed. Set to `True` to remove extra spaces, or `False` to keep them.

By setting these flags to either `True` or `False`, you can customize the preprocessing steps according to your requirements. The code applies the specified preprocessing steps to each text element in the series or column and returns the processed text.




**Text preprocessing sample code**

```python

from mlfast import Text_preprocessing


df['review'].apply(Text_preprocessing,
                  stem=False,
                  lemmatize=True,
                  remove_html=True,
                  remove_emoji=True,
                  remove_special_chars=True,
                  remove_extra_spaces=True)

```



### Chatbot



Import the `Chatbot` class: In your Python script, import the `Chatbot` class from `mlfast`. You can do this by adding the following line at the beginning of your code:


```python
from mlfast import Chatbot
```

- Obtain an OpenAI API key: To use the `mlfast` library, you'll need an API key from OpenAI. If you don't have one, sign up on the OpenAI website and obtain an API key.

- Create a `Chatbot` instance: Initialize the Chatbot class with your OpenAI API key and specify the desired role for your chatbot. Here's an example of creating a `Chatbot` instance:

- Replace `"YOUR-OPENAI-API-KEY"` with your actual OpenAI API key, and `"ENTER-YOUR-CHATBOT-ROLE"` with the desired role for your chatbot.

- Deploy the chatbot: Set the `deploy` parameter to `True` when creating the `Chatbot` instance. This will deploy the chatbot and make it available for use.

- To deploy or terminate the deployment of a chatbot created using the `mlfast` library, you can set the `deploy` parameter to `True` or `False` when creating the `Chatbot` instance, respectively.


```python
Chatbot(api_key="YOUR-OPENAI-API-KEY",
        role="ENTER-YOUR-CHATBOT-ROLE",
        deploy=True)
```
# **Chatbot feature is depreciated will be added soon on different library**



## Announcement

- Unsupervised Machine Learning Algorithms
- Bag of words, TFIDF and Word2Vec
- Image Preprocessing
- And many more



**ADDED SOON**