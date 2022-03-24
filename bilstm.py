import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
import numpy as np
import pandas as pd
from termcolor import colored

# Load data
print(colored("Loading train and test data", "yellow"))
train_data = pd.read_csv(r'C:\Users\Msı\Desktop\YL\Derin_Ogrenme\Twitter-Sentiment-Analysis\train.csv')
test_data = pd.read_csv(r'C:\Users\Msı\Desktop\YL\Derin_Ogrenme\Twitter-Sentiment-Analysis\test.csv')
print(colored("Data loaded", "yellow"))
class_len= len(train_data['sentiment-class'].unique())
print(len(test_data['sentiment-class'].unique()))
# Tokenization
print(colored("Tokenizing and padding data", "yellow"))
tokenizer = Tokenizer(num_words = 2000, split = ' ')
tokenizer.fit_on_texts(train_data['tweet'].astype(str).values)
train_tweets = tokenizer.texts_to_sequences(train_data['tweet'].astype(str).values)
max_len = max([len(i) for i in train_tweets])
train_tweets = pad_sequences(train_tweets, maxlen = max_len)
test_tweets = tokenizer.texts_to_sequences(test_data['tweet'].astype(str).values)
test_tweets = pad_sequences(test_tweets, maxlen = max_len)
print(colored("Tokenizing and padding complete", "yellow"))

# Building the model


y_train = np.asarray(train_data['sentiment-class'].values).astype('float32').reshape((-1,1))
y_test = np.asarray(test_data['sentiment-class'].values).astype('float32').reshape((-1,1))



embedding_layer = Embedding(2000,
                            128,
                            input_length=43,
                            )

model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
    Conv1D(100, 10, activation='relu'),
    GlobalMaxPool1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
],
name="Sentiment_Model")



#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ['accuracy'])
model.summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=100, cooldown=0),
#             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100)]

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
   train_tweets,
   y_train,
    batch_size=1024,
    epochs=100,
    validation_split=0.1,
    #callbacks=callbacks,
    verbose=1,
)


acc,  val_acc  = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()