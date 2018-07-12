
# coding: utf-8

# In[ ]:


from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
import h5py
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, Merge, Conv2D, MaxPooling2D, Flatten, Dense
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import keras
import gensim
import pickle
from tqdm import tqdm


# In[ ]:


# loading the hdf5 file
f1 = h5py.File('../objects/data.h5','r+')


# In[ ]:


#train and test sizes
x1 = f1['X_train'].shape[0]
x2 = f1['X_train'].shape[1]
x3 = f1['X_test'].shape[0]
x1 , x2, x3


# In[ ]:


# loading matrices for training and testing
X_train = np.reshape(f1['X_train'][:], (x1, x2, x2, 1))
Y_train = f1['Y_train'][:]
X_test = np.reshape(f1['X_test'][:], (x3, x2, x2, 1))
Y_test = f1['Y_test'][:]


# In[ ]:


# CNN model
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (x2, x2, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(x=X_train, y=Y_train, steps_per_epoch=x1, validation_split=0.2, epochs=25)


# In[ ]:


classifier.evaluate(X_test, Y_test, verbose=1)


# In[ ]:


classifier.metrics_names


# In[ ]:


#classifier.predict(X_test)

