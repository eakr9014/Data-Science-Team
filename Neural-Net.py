import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer

# read in train and test data
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# tf-idf on data
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(train["text"])
vectors_test = vectorizer.transform(test["text"])
y_train = train["target"]
vectors_train, y_train = shuffle(vectors_train, y_train)

# print shapes and divide to test/validate
rows = vectors_train.shape[0]
columns = vectors_train.shape[1]
# print("Shape of tf-idf", vectors_train.shape)
# x_val = vectors_train[:int(rows/5)].toarray()
# print("x-val shape", x_val.shape)
# y_val = y_train[:int(rows/5)]
# print("y-val shape", y_val.shape)
# x_trn = vectors_train[int(rows/5):].toarray()
# print("\nx-trn shape", x_trn.shape)
# y_trn = y_train[int(rows/5):]
# print("y-trn shape", y_trn.shape)
x_trn = vectors_train[:].toarray()
print("x-trn shape", x_trn.shape)
y_trn = y_train[:]
print("y-trn shape", y_trn.shape)

#test data
test_x = vectors_test.toarray()

#create model
print("\nbefore training")
model = tf.keras.Sequential()
model.add(layers.Dense(columns//4, activation='relu', input_shape=(columns,)))
model.add(layers.Dropout(.4))
model.add(layers.Dense(columns//2, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(columns//8, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(columns//4, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(1, activation='sigmoid'))

# x_trn = x_trn[:,:,None]
# x_val = x_val[:,:,None]
# model.add(layers.LSTM(columns, input_shape=(columns,1)))
# model.add(layers.Dense(columns//8, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

#compile and fit model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print("\nafter compile\n")
# history = model.fit(x_trn, y_trn, epochs=3, validation_data=(x_val,y_val), verbose=1, batch_size=512)
history = model.fit(x_trn, y_trn, epochs=3, verbose=1, batch_size=512)

predictions = model.predict(test_x)
binary = []
for predict in predictions:
    if predict > .5:
        binary.append(1)
    else:
        binary.append(0)

idxs = test["id"]
dict = {"id" : idxs, "target" : binary}
pd.DataFrame(dict).to_csv("answers.csv", index=False)
