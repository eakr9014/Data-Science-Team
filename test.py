import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import csv


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])

clf = MultinomialNB()
clf.fit(train_vectors, train_df["target"]);

sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()

with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "target"])
    for i in range(len(test_df.index)):
        row = [test_df["id"][i], clf.predict(test_vectors[i])[0]]
        writer.writerow(row)

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
print(scores)