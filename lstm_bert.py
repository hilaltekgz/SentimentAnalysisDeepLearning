import json
import random
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer



import os
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dropout
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
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
device = 'cuda'
print(colored("Loading train and test data", "yellow"))
train_data = pd.read_csv(r'C:\Users\Msı\Desktop\YL\Derin_Ogrenme\Twitter-Sentiment-Analysis\train.csv')
test_data = pd.read_csv(r'C:\Users\Msı\Desktop\YL\Derin_Ogrenme\Twitter-Sentiment-Analysis\test.csv')
print(colored("Data loaded", "yellow"))
class_len= len(train_data['sentiment-class'].unique())
print(len(test_data['sentiment-class'].unique()))
# Tokenization




bert = AutoTokenizer.from_pretrained('bert-base-uncased')




train = list(train_data['tweet'])

print(colored("Tokenizing and padding data", "yellow"))

train_tweets = bert(train, truncation=True, padding=True)
max_len = max([len(i) for i in train_tweets])
print('max_length', max_len)
#train_tweets = pad_sequences(train_tweets, maxlen = max_len)
test_tweets = bert(test_data['tweet'].to_list(), truncation=True, padding=True)
#test_tweets = pad_sequences(test_tweets, maxlen = max_len)
print('test tweet',type( test_tweets))
print(colored("Tokenizing and padding complete", "yellow"))
#print(train_tweets.shape[0])
#print(train_tweets.shape[1])




def feature_extraction(text):
    x = bert.encode(filter(text))
    with torch.no_grad():
        x, _ = torch.stack([torch.tensor(x)])
        return list(x[0][0].cpu().numpy())

print(colored("Creating the LSTM model", "yellow"))
model = Sequential()
model.add(Embedding(2000, 128, input_length = train_tweets))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(64, dropout = 0.4))
model.add(Dense(class_len, activation = 'softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics = ['accuracy'])
model.summary()
from tensorflow.keras.utils import to_categorical
y_test =  to_categorical(test_data['sentiment-class'].values)
# Training the model
print(colored("Training the LSTM model", "green"))

#callback= EarlyStopping(monitor='val_loss')

history = model.fit(train_tweets, pd.get_dummies(train_data['sentiment-class']).values, epochs = 1, batch_size = 128, validation_split = 0.1)
print(colored(history, "green"))

# Testing the model
print(colored("Testing the LSTM model", "green"))
score, accuracy = model.evaluate(test_tweets, pd.get_dummies(test_data['sentiment-class']).values, batch_size = 128)
print("Test accuracy: {}".format(accuracy))
