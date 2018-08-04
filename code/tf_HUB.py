
# coding: utf-8

# In[ ]:


from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Embedding, LSTM, Merge, Activation, Dense, Conv1D, MaxPooling1D, AveragePooling1D, GRU, Dropout, TimeDistributed
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# In[ ]:


TRAIN_CSV = '../dataset/train.txt'
TEST_CSV = '../dataset/test.txt'
EMBEDDING_FILE = '../models/GoogleNews-vectors-negative300.bin'
MODEL_SAVING_DIR = '../models/'


# In[ ]:


file1 = open(TRAIN_CSV,"r")
data = []
for line in file1:
    fields = line.split("\t")
    data.append(fields)
a = []
b = []
c = []
d = []
e = []
for element in data:
    a.append(element[0])
    b.append(element[1])
    c.append(element[2])
    d.append(element[3])
    e.append(element[4])
print(type(a[0]))
d = {'Quality': a[1:], 'sentence1': d[1:], 'sentence2': e[1:]}
train_df = pd.DataFrame(data=d)

file2 = open(TEST_CSV,"r")
data = []
for line in file2:
    fields = line.split("\t")
    data.append(fields)
a = []
b = []
c = []
d = []
e = []
for element in data:
    a.append(element[0])
    b.append(element[1])
    c.append(element[2])
    d.append(element[3])
    e.append(element[4])
print(type(a[0]))
d = {'Quality': a[1:], 'sentence1': d[1:], 'sentence2': e[1:]}
test_df = pd.DataFrame(data=d)


# In[ ]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)


# In[ ]:


module_path = "../universal_Sentence_encoder/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/"
#module_path = "https://tfhub.dev/google/universal-sentence-encoder-large/3"


# In[ ]:


embed = hub.Module(module_path, trainable = False)


# In[ ]:


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    if c2 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2

    return precision

def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall = c1 / c3

    return recall

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# In[ ]:


#messages = ["That band rocks!", "That song is really cool."]

#with tf.Session() as session:
#  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#  message_embeddings = session.run(embed(messages))
#message_embeddings


# In[ ]:


sentence_cols = ['sentence1', 'sentence2']

X = train_df[sentence_cols]
Y = train_df['Quality']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20)

# Split to dicts
X_train = {'left': np.array(X_train['sentence1'].tolist(), dtype=object)[:, np.newaxis], 'right': np.array(X_train['sentence2'].tolist(), dtype=object)[:, np.newaxis]}
X_test = {'left': np.array(test_df['sentence1'].tolist(), dtype=object)[:, np.newaxis], 'right': np.array(test_df['sentence2'].tolist(), dtype=object)[:, np.newaxis]}
X_validation = {'left': np.array(X_validation['sentence1'].tolist(), dtype=object)[:, np.newaxis], 'right': np.array(X_validation['sentence2'].tolist(), dtype=object)[:, np.newaxis]}

Y_test = test_df['Quality']

Y_train = Y_train.values
Y_test = Y_test.values
Y_validation = Y_validation.values

#train_text = np.array(train_text, dtype=object)[:, np.newaxis]

left_input = Input(shape=(1,), dtype='int32')
right_input = Input(shape=(1,), dtype='int32')


# In[ ]:


#X_train['left'] = np.reshape(X_train['left'],(X_train['left'].shape[0], 1, X_train['left'].shape[1]))
#print(X_train['left'].shape)


# In[ ]:


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


# In[ ]:


from keras.layers import Flatten
left_input = Input(shape=(1,), dtype=tf.string)
left_embedding = Lambda(UniversalEmbedding, output_shape=(512,))(left_input)
#left_conv1 = Conv1D(128, 7, activation='relu', strides=1)(left_embedding)
#left_pool1 = l_pool_shared1 = AveragePooling1D(7, strides=1)(left_conv1)
left_dense1 = Dense(256, activation='relu')(left_embedding)
left_dense2 = Dense(128, activation='softmax')(left_dense1)
left_dense3 = Dense(1, activation='sigmoid')(left_dense2)
#left_dense4 = Dense(32, activation='softmax')(left_dense3)
#left_dense5 = Dense(1, activation='softmax')(left_dense4)
#left_lstm = LSTM(50, return_sequences=True)(left_dense2)
#left_flatten = Flatten()(left_dense2)
#left_activation = Activation('softmax')(left_flatten)
#left_vec = RepeatVector(50)(left_activation)
#left_permute = Permute([2, 1])(left_vec)
model1 = left_dense3


# In[ ]:


print(left_embedding.shape)
print(left_dense1.shape)
print(left_dense2.shape)
#print(left_lstm.shape)


# In[ ]:


right_input = Input(shape=(1,), dtype=tf.string)
right_embedding = Lambda(UniversalEmbedding, output_shape=(512,))(right_input)
#right_conv1 = Conv1D(128, 7, activation='relu', strides=2)(right_embedding)
#right_pool1 = l_pool_shared1 = AveragePooling1D(7, strides=2)(right_conv1)
right_dense1 = Dense(256, activation='relu')(right_embedding)
right_dense2 = Dense(128, activation='softmax')(right_dense1)
right_dense3 = Dense(1, activation='sigmoid')(right_dense2)
#right_dense4 = Dense(32, activation='softmax')(right_dense3)
#right_dense5 = Dense(1, activation='softmax')(right_dense4)
#right_lstm = LSTM(50, return_sequences=True)(right_dense2)
model2 = right_dense3


# In[ ]:


print(right_dense2.shape)


# In[ ]:


from keras.layers import merge
#model1 = merge([left_dense1, left_dense2], mode='concat')
#model2 = merge([right_dense1, right_dense2], mode='concat')


# In[ ]:


print(model1.shape)


# In[ ]:


malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([model1, model2])


# In[ ]:


malstm = Model([left_input, right_input], [malstm_distance])


# In[ ]:


from keras import optimizers

gradient_clipping_norm = 1.25
optimizer = optimizers.Adadelta(lr=0.8, clipnorm=gradient_clipping_norm)

malstm.compile(loss='mse',
              optimizer=optimizer,
              metrics=['accuracy', precision, recall, f1_score])
#malstm.summary()


# In[ ]:


with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = malstm.fit([X_train['left'], X_train['right']], 
            Y_train,
            validation_data=([X_validation['left'], X_validation['right']], Y_validation),
            epochs=5,
            batch_size=64)
  evaluate = malstm.evaluate([X_test['left'], X_test['right']], Y_test, verbose=1)
  print(evaluate)
  malstm.save_weights('../objects/model.h5')

