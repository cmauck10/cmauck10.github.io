---
title: "Improving Model Performance with Cleanlab"
date: 2022-08-26T11:08:04-05:00
draft: false
---
# Improving Model Performance with Cleanlab
## Can we improve performance on baseline models?

To offer some inspiration for this, the data that I used comes from a project I did my junior year at MIT. A group of friends and I took it upon ourselves to try and "beat" Vegas at predicting NFL game outcomes. Using their "line" to determine which team they believed would win, we used our classification model (a GBC) to make our predictions. To save you time from reading the non-existent write-up, we ended up tying them on the 2018 season and beating them by 3% on the 2019 seaosn.

As I revisit my project from years ago, I have a new tool in my arsenal and a new trick up my sleeve. Using a nifty wrapper from the open source project [Cleanlab](https://docs.cleanlab.ai/v2.0.0/index.html%20), I can now use any sk-learn styled model, at a presumably higher accuracy. By using their wrapper, I'm tapping into their [black magic](https://github.com/cleanlab/cleanlab/discussions/56#discussioncomment-358124) that identifies noise in my training set's labels, and trains the selected model after removing labels that exceed a certain noise threshold. 

Let's take a look at a few different classes of models and see if the added functionality improves our accuracy. 

## Imports


```python
# %pip install lightgbm
# %pip install xgboost
# %pip install catboost
# %pip install lightgbm
# %pip install cleanlab
import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

#future warning was being annoying
import warnings
warnings.filterwarnings('ignore')
```

## Data
Our data consists of week-by-week data from 2002 until the 2019 NFL season.   


```python
#import our football statistics data
df = pd.read_csv("master-boxscore-tracker.csv")
#show our data
df.head()
#note that data shown below is a small subset of our data
```

| Season | Week | Home Team | Home Score | Away Team | Away Score | Home Win | … | Home Total Yards | Home Yards Allowed | Away Total Yards |
|--------|------|-----------|------------|-----------|------------|----------|---|------------------|--------------------|------------------|
| 2002   | 2    | CLE       | 20         | CIN       | 7          | TRUE     | … | 411              | 470                | 203              |
| 2002   | 2    | IND       | 13         | MIA       | 21         | FALSE    | … | 307              | 343                | 389              |
| 2002   | 2    | DAL       | 21         | TEN       | 13         | TRUE     | … | 267              | 210                | 328              |
| 2002   | 2    | CAR       | 31         | DET       | 7          | TRUE     | … | 265              | 289                | 257              |
| 2002   | 2    | BAL       | 0          | TB        | 25         | FALSE    | … | 289              | 265                | 333              |


## Data Pre-processing and Train/Test Split
Before we can test our models, we need to get our data into a format they can handle. Here, we convert our label columns to 1's and 0's, as well as slice our data to only include the numerical data we want to train our models on. We also split our data into our training and testing sets.


```python
#convert T/F col to 1/0
df["Home Win"] =  df["Home Win"].astype(int)
#only want numerical data
X = np.array(df.loc[:, "Line":])
#get labels
y = np.array(df['Home Win'])
#we will use default split for now 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
```

## Model Selection
Now let's see how this adaptation class does when used with a variety of different models.
- Basic 
    - KNN
    - SVM
    - MLP 
- Ensemble 
    - RandomForest
- Boosting
    - LightGBM
    - XGBoost
    - CatBoost

```python
#models we will be using with CL wrapper
models = [
    #basic models
    KNeighborsClassifier(),
    SVC(probability = True),
    MLPClassifier(),

    #ensemble model(s)
    RandomForestClassifier(), 

    #boosting models
    LGBMClassifier(), 
    XGBClassifier(), 
    CatBoostClassifier(silent=True),   
]

model_names = [type(model).__name__ for model in models]
```

## Model Evaluation
To utilize the Cleanlab wrapper, we simply use

`clf = SomeSklearnClassifier()`

`model = CleanLearning(clf=clf)`

You can then use any of the sk-learn methods on the `CleanLearning`object. 


```python
def test_clf(model, model_name):
  #iniiate our models
  clf = model
  clf_cl = CleanLearning(clf=clf)

  #fit baseline model
  clf.fit(X_train, y_train)
  pred = clf.predict(X_test)
  clf_acc = accuracy_score(y_test, pred)
  clf_pct = "{:.2%}".format(clf_acc)

  #fit baseline model with Cleanlab wrapper
  clf_cl.fit(X_train, y_train)
  pred = clf_cl.predict(X_test)
  clf_cl_acc = accuracy_score(y_test, pred)
  clf_cl_pct = "{:.2%}".format(clf_cl_acc)

  #get difference in model perf
  delta = clf_cl_acc-clf_acc
  delta_pct = "{:.2%}".format(delta)

  #print results
  print("{} accuracy: {}".format(model_name, clf_pct))
  print("{} w/ cl accuracy: {}".format(model_name, clf_cl_pct))
  print("Cleanlab improvement: {}".format(delta_pct))
  print("---------------------------------------------")
```


```python
for (model, model_name) in zip(models, model_names):
  test_clf(model, model_name)
```

    KNeighborsClassifier accuracy: 56.45%
    KNeighborsClassifier w/ cl accuracy: 60.63%
    Cleanlab improvement: 4.19%
    ---------------------------------------------
    SVC accuracy: 57.87%
    SVC w/ cl accuracy: 63.00%
    Cleanlab improvement: 5.13%
    ---------------------------------------------
    MLPClassifier accuracy: 56.72%
    MLPClassifier w/ cl accuracy: 58.95%
    Cleanlab improvement: 2.23%
    ---------------------------------------------
    RandomForestClassifier accuracy: 64.62%
    RandomForestClassifier w/ cl accuracy: 66.10%
    Cleanlab improvement: 1.49%
    ---------------------------------------------
    LGBMClassifier accuracy: 64.48%
    LGBMClassifier w/ cl accuracy: 65.50%
    Cleanlab improvement: 1.01%
    ---------------------------------------------
    XGBClassifier accuracy: 61.31%
    XGBClassifier w/ cl accuracy: 64.42%
    Cleanlab improvement: 3.11%
    ---------------------------------------------
    CatBoostClassifier accuracy: 65.09%
    CatBoostClassifier w/ cl accuracy: 65.02%
    Cleanlab improvement: -0.07%
    ---------------------------------------------


## Results
We see considerable increases in some of the models. These deltas change on each iteration quite significantly, so further testing will be necessary to determine the average change that the cl wrapper produces. We can say, however, with confidence that the added "black magic" does in fact increase performance on baseline models. By reducing noise in the input data, most our models are able to predict at higher accuracies. It's also important to note that all of these models are run with default hyperparameters. Further work would need to be done in order to tune each model individually and then apply the cl wrapper to determine if the same increase deltas exist.    

## Further Work
- Tune each model to its optimal state (grid search for hyperparams) and then check delta with CL wrapper
- Include visualizations for models
- Include more classification models
- Try with non-tabular data such as image or text data (VGG16, ResNet50, GPT3, etc.) 
- Additional written content
- Data pre-processing like PCA or variance analysis
