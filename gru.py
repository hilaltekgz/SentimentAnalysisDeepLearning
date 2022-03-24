import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
print(colored("Creating the GRU model", "yellow"))

EMBEDDING_DIM = 100
vocab_size = 2000
print('Build model...')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=train_tweets.shape[1]))
model.add(GRU(units=32,  dropout=0.2))
model.add(Dense(class_len, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics = ['accuracy'])

print('Summary of the built model...')
print(model.summary())


from tensorflow.keras.utils import to_categorical
y_test =  to_categorical(test_data['sentiment-class'].values)
# Training the model
print(colored("Training the LSTM model", "green"))
history = model.fit(train_tweets, pd.get_dummies(train_data['sentiment-class']).values, epochs = 50, batch_size = 128, validation_split = 0.2)
print(colored(history, "green"))

# Testing the model
print(colored("Testing the LSTM model", "green"))
score, accuracy = model.evaluate(test_tweets, pd.get_dummies(test_data['sentiment-class']).values, batch_size = 128)
print("Test accuracy: {}".format(accuracy))  

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

