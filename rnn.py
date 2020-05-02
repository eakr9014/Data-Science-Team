
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
Y_train = train["target"]
X_train = train["text"]
x_test = test["text"]
# X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.15, stratify=y_train, shuffle=True)

max_words = 10000
max_len = 1200
tok = Tokenizer(num_words=max_words, lower=True)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
# val_sequences = tok.texts_to_sequences(X_val)
# val_sequences_matrix = sequence.pad_sequences(val_sequences,maxlen=max_len)

inputs = Input(name='inputs',shape=[max_len])
layer = Embedding(max_words,128,input_length=max_len)(inputs)
layer = Bidirectional(LSTM(128), name='BD_LSTM')(layer)
layer = Dense(256,name='FF1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(64,name='FF2')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,name='out_layer')(layer)
layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(sequences_matrix, Y_train, batch_size=256, epochs=3, validation_data=(val_sequences_matrix, Y_val))
history = model.fit(sequences_matrix, Y_train, batch_size=256, epochs=3)

test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

predictions = model.predict(test_sequences_matrix)
binary = []
for predict in predictions:
    if predict > .5:
        binary.append(1)
    else:
        binary.append(0)

idxs = test["id"]
print(len(idxs), len(predictions))
dict = {"id" : idxs, "target" : binary}
pd.DataFrame(dict).to_csv("answers.csv", index=False)