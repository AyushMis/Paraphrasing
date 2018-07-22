
# coding: utf-8

# In[ ]:


from time import time
import pandas as pd
import numpy as np
import h5py
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import keras
import gensim
import pickle
from tqdm import tqdm


# In[ ]:


# path of dataset
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


stops = set(stopwords.words('english'))

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

sentence_cols = ['sentence1', 'sentence2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for sentence in sentence_cols:

            s2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[sentence]):

                # Check for unwanted words
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    s2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    s2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, sentence, s2n)
            
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


# In[ ]:


print(len(test_df))
print(len(train_df))


# In[ ]:


# determining maximum length of sentence
max_seq_length = max(train_df.sentence1.map(lambda x: len(x)).max(),
                     train_df.sentence2.map(lambda x: len(x)).max(),
                     test_df.sentence1.map(lambda x: len(x)).max(),
                     test_df.sentence2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 1000
training_size = len(train_df) - validation_size

X_train = train_df[sentence_cols]
Y_train = train_df['Quality']

# Split to dicts
X_train = {'left': X_train.sentence1, 'right': X_train.sentence2}
X_test = {'left': test_df.sentence1, 'right': test_df.sentence2}
Y_test = test_df['Quality']

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_test = Y_test.values

# Zero padding
for dataset, side in itertools.product([X_train, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# In[ ]:


# determining size of dataset
x1 = X_train['left'].shape[0]
x2 = X_train['left'].shape[1]
similarity_matrix_train = np.zeros(((x1, x2, x2)))
print(similarity_matrix_train.shape)


# In[ ]:


# norm2 by row
norm2 = lambda X : K.expand_dims(K.sqrt(K.sum(X ** 2, 1)))
# Cosine distance by its best definition#
cosine = lambda X, Y : K.dot(X, K.transpose(Y))/norm2(X)/K.transpose(norm2(Y))


# In[ ]:


# Similarity matrix for training data
train_sim = np.ndarray(shape=(x1, x2, x2), dtype=float)
for i in tqdm(range(x1)):
    emb1 = embeddings[X_train['left'][i]]
    emb2 = embeddings[X_train['right'][i]]
    emb1 = tf.convert_to_tensor(emb1.astype(float), np.float64)
    emb2 = tf.convert_to_tensor(emb2.astype(float), np.float64)
    x = cosine(emb1, emb2)
    with tf.Session():
        train_sim[i] = x.eval()


# In[ ]:


x3 = X_test['left'].shape[0]
x4 = X_test['left'].shape[1]
test_sim = np.ndarray(shape=(x3, x4, x4), dtype=float)


# In[ ]:


# Similarity matrix for testing data
for i in tqdm(range(x3)):
    emb1 = embeddings[X_test['left'][i]]
    emb2 = embeddings[X_test['right'][i]]
    emb1 = tf.convert_to_tensor(emb1.astype(float), np.float64)
    emb2 = tf.convert_to_tensor(emb2.astype(float), np.float64)
    x = cosine(emb1, emb2)
    with tf.Session():
        test_sim[i] = x.eval()


# In[ ]:


# Storing all the matrices in hdf5 files
hf = h5py.File('../objects/data.h5', 'w')
hf.create_dataset('X_train', data=train_sim)
hf.create_dataset('X_test', data=test_sim)
hf.create_dataset('Y_test', data=Y_test)
hf.create_dataset('Y_train', data=Y_train)


# In[ ]:


hf.close()
