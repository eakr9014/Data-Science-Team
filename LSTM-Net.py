import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
import keras

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

model = keras.Sequential()
model.add(layers.LSTM(128))