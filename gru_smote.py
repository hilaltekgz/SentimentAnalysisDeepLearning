import os
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dropout
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored
from tensorflow.keras.callbacks import EarlyStopping
# Load data
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split

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
print('max_length', max_len)
train_tweets = pad_sequences(train_tweets, maxlen = max_len)
test_tweets = tokenizer.texts_to_sequences(test_data['tweet'].astype(str).values)
test_tweets = pad_sequences(test_tweets, maxlen = max_len)
print('test tweet', test_tweets)
print(colored("Tokenizing and padding complete", "yellow"))
print(train_tweets.shape[0])
print(train_tweets.shape[1])




#train oversampling 
counter_test = Counter(train_data['sentiment-class'].values)
print('Oversampling öncesi',counter_test)
#Oversampling
ros = RandomOverSampler(random_state=777)
train_tweets, y_train = ros.fit_sample(train_tweets, train_data['sentiment-class'].values)
#train_tweets = train_tweets.todense()
print(y_train)
counter_test = Counter(y_train)
print('Oversampling sonrası',counter_test)
#Oversampling
print('shape',train_tweets.shape)





counter_test = Counter(test_data['sentiment-class'].values)
print('Oversampling öncesi',counter_test)
#Oversampling
ros = RandomOverSampler(random_state=777)
test_tweets, y_ROS = ros.fit_sample(test_tweets, test_data['sentiment-class'].values)
print(y_ROS)
#test_tweets = test_tweets.todense()
counter_test = Counter(y_ROS)
print('Oversampling sonrası',counter_test)
#Oversampling

print(train_tweets.shape[0])
print(train_tweets.shape[1])
from tensorflow.keras.utils import to_categorical
y_test =  to_categorical(y_ROS,3)
y_train =  to_categorical(y_train,3)



X_train, X_val, Y_train, Y_val = train_test_split(train_tweets, y_train, test_size = 0.20, random_state = 100)
EMBEDDING_DIM = 100
vocab_size = 2000

# Building the model
print('Build model...')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=train_tweets.shape[1]))
model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(class_len, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics = ['accuracy'])
model.summary()
#y_test =  to_categorical(test_data['sentiment-class'].values)
# Training the model
print(colored("Training the LSTM model", "green"))

#callback= EarlyStopping(monitor='val_loss') 

history = model.fit(X_train, Y_train, epochs = 30, batch_size = 128, validation_data=(X_val, Y_val))
print(colored(history, "green"))

# Testing the model
print(colored("Testing the LSTM model", "green"))
score, accuracy = model.evaluate(test_tweets, y_test, batch_size = 128)
print("Test accuracy: {}".format(accuracy))

#import numpy as np
#from sklearn.metrics import confusion_matrix
#from sklearn import metrics
#test_labels = to_categorical(pd.get_dummies(test_data['sentiment-class']).values)
#print('test_labels',test_labels)
#y_pred = model.predict(test_tweets)
###y_pred=np.argmax(y_pred,axis=1)
#print('y_pred',y_pred)
##print('Karmaşıklık Matrisi',confusion_matrix(test_labels, y_pred))
#print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
#print('F1 Score:',metrics.f1_score(test_labels, y_pred, average='weighted'))
#print('AUC:',metrics.roc_auc_score(test_labels, y_pred))

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


