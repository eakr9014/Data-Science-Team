# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")
x=train_df.drop(columns=["target"], axis=1)
y=train_df["target"]
X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.25,random_state=42)

count_vectorizer = feature_extraction.text.CountVectorizer()

## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(X_train["text"][0:5])

train_vectors = count_vectorizer.fit_transform(X_train["text"])

val_vectors= count_vectorizer.transform(X_val["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors -
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])

y=train_df["target"]
#y_test=test_df["target"]

## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression
## is a good way to do this.

param_grid={'n_estimators' : [150,200,250,300]}

clf = RandomForestClassifier(n_estimators=150)
# check = GridSearchCV(clf, param_grid, cv=3);
# check.fit(train_vectors, y)


# print(check.best_estimator_)  #Gives parameters that minimize the lost function the best
# print("Best parameter is ", check.best_params_)
# print(check.cv_results_)

#clf = RandomForestClassifier(n_estimators=300)

clf.fit(train_vectors, y_train)
sample_submission = pd.read_csv("sample_submission.csv")
y_pred= clf.predict(val_vectors)
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
#
# scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
# print(scores)

print('accuracy score: ',accuracy_score(y_pred,y_val))
print(classification_report(y_val, y_pred))