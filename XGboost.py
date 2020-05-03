import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def join_clean(text):
    text = text.lower()
    text = [letter for letter in text.split() if len(letter) > 2] #split each letter
    text = ' '.join(text) #make all into one long word
    return text

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load i

def plot_words(word_dict):

    dict_tuples=[(v,k) for k,v in word_dict.items()]
    #print(len(dict_tuples), type(dict_tuples))
    dict_tuples.sort(key = lambda x: x[0], reverse=True)
    dict_tuples= dict_tuples[0:15]
    #print(dict_tuples)
    y_vals, x_vals = zip(*dict_tuples)
    #print(y_vals)
    x_vals= np.arange(len(y_vals))
    labels= word_dict.keys()
    plt.bar(x_vals, y_vals)
    plt.xticks(x_vals, labels)
    plt.show()



def count_words(tweet_list):
    d={}
    for tweet in tweet_list:
        tweet = tweet.strip()
        # Convert the characters in line to
        # lowercase to avoid case mismatch
        tweet = tweet.lower()

        # Split the line into words
        words = tweet.split(" ")

        # Iterate over each word in line
        for word in words:
            # Check if the word is already in dictionary
            if(len(word)<4):
                continue;
            if word in d:
                # Increment count of word by 1
                d[word] = d[word] + 1
            else:
                # Add the word to dictionary with count 1
                d[word] = 1
    return d






def main():

    train_df = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")




    # Removing null values in train
    train = train_df.dropna()

    #
    train_len = len(train)
    print(train_len, len(train))
    test_len = len(test)
    target = train["target"].values
    train.drop(['target'], axis=1)
    df_combined = pd.concat((train, test)) # After cleaning both datasets combine
    #print("df_combined size is : {}".format(df_combined.shape))

    df_combined = df_combined.dropna() #get rid of any columns with missing data for our clean text



    df_combined['Keyword_clean'] = df_combined['keyword'].apply(join_clean)
    df_combined['Text_clean'] = df_combined['text'].apply(join_clean)



    df_combined['Key'] = df_combined['Keyword_clean'] +' ' + df_combined['Text_clean']
    #df_combined['Key_Text'] = df_combined['keyword'] +' ' + df_combined['text']



    # Resplit the data
    train_df = df_combined[0:train_len]
    test_df = df_combined[test_len:]



    X_train, X_test, y_train, y_test = train_test_split(train_df['Text_clean'], target, test_size=0.25, stratify=target)

    tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None)

    X_train = tfidf_vect.fit_transform(X_train)
    X_test= tfidf_vect.transform(X_test)

    x_model = XGBClassifier(learning_rate = 0.08, objective = 'binary:logistic', n_estimators = 150)
    x_model.fit(X_train, y_train)
    y_pred = x_model.predict(X_test)
    train_pred = x_model.predict(X_train)



    # index= [i for i in range(len(y_pred)) if y_pred[i]]
    #
    # #print("index is", index )
    # text_val= X_test["text"].tolist()
    #
    # true_tweets= [text_val[i] for i in index]
    #
    # word_dict=count_words(true_tweets)
    # plot_words(word_dict)

    print('Train accuracy score :', accuracy_score(y_train, train_pred))
    #print('Test accuracy score :', accuracy_score(y_test, y_pred))
    print(classification_report(y_train, train_pred))






main()
