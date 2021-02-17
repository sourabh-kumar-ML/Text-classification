#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import numpy as np
import gensim

import nltk as nl
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,OPTICS,SpectralClustering

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.layers as layers


# In[2]:


#Reading datasets

data_train = pd.read_csv("train_extracted.csv")
data_test = pd.read_csv("test_extracted.csv")


# In[3]:


data_train.head()


# In[41]:


# Splitting sentences into words

train_tokens = [nl.word_tokenize(sentences) for sentences in data_train.text]
test_tokens = [nl.word_tokenize(sentences) for sentences in data_test.text]


# In[42]:


#training word2vec model on complete vocabulary (train + test vocabulary)

model = gensim.models.Word2Vec(size=100, min_count=1, workers=-1)
model.build_vocab((train_tokens+test_tokens))
model.train((train_tokens+test_tokens),total_examples= len(train_tokens+test_tokens),epochs = 2500)


# In[43]:


# storing trained values of every word [-1,100] --> [-1,1] 
w2v = dict(zip(model.wv.index2word, np.mean(model.wv.syn0,axis=1)))


# In[86]:


# Function to map words to their respective values 
def prepare_text(tokens):
    text = []
    max_len = 0
    for sentence in tokens:
        line = []
        for word in sentence:
            try:
                line.append(w2v[word])
            except:
                print(word)
                line.append(0)
        max_len = max(max_len,len(line))       
        text.append(line)
    return np.array(text),max_len


# In[87]:


# Converting strinf to float values and getting max_len for padding
X_train,max_len = prepare_text(train_tokens)
X_test,_ = prepare_text(test_tokens)


# In[88]:


# Padding to ensure dimensions

X_train = pad_sequences(X_train,dtype='float32',padding='post',maxlen= max_len)
X_test = pad_sequences(X_test,dtype='float32',padding='post',maxlen = max_len )


# In[89]:


# converting labels to one-hot-vector and categorical-vector
oe_enc = OneHotEncoder()

Y_train = oe_enc.fit_transform(np.array(data_train.label).reshape(-1,1)).toarray()
Y_test = oe_enc.fit_transform(np.array(data_test.label).reshape(-1,1)).toarray()

true_label_train = np.argmax(Y_train,axis=1)
true_label_test = np.argmax(Y_test,axis=1)


# In[90]:


# storing true dimensions
n_len,features = X_train.shape
n_len_test,features = X_test.shape


# In[91]:


# defining optimizer and callback
optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)


# # RNN

# In[92]:


model = Sequential(name="RNN")
model.add(layers.Embedding(input_dim=X_train.shape[1], output_dim=64))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(6,activation="softmax"))


# In[93]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])


# In[94]:


model.fit(X_train,Y_train,validation_data = (X_test,Y_test),callbacks=[es],epochs=50)


# # LSTM

# In[95]:


X_train,X_test = X_train.reshape(n_len,features,1),X_test.reshape(n_len_test,features,1)


# In[97]:


model = Sequential(name="LSTM")
model.add(layers.LSTM(128,input_shape=(X_train.shape[1],1)))
model.add(layers.Dense(6,activation="softmax"))


# In[98]:


model.compile(optimizer = optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[99]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),callbacks=[es],epochs=50,verbose=1)


# # Bi-LSTM

# In[103]:


model=Sequential(name="Bi-LSTM")
model.add(layers.Bidirectional(layers.LSTM(64,input_shape=(X_train.shape[1],1),activation="relu")))
model.add(layers.Dense(6,activation="softmax"))


# In[104]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])


# In[106]:


model.fit(X_train,Y_train,validation_data = (X_test,Y_test),callbacks=[es],epochs=50)


# # CNN

# In[107]:


model = Sequential(name = "CNN")
model.add(layers.Conv1D(filters=32,kernel_size=4,input_shape=(X_train.shape[1],1),activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(filters=64,kernel_size=6,activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Dense(6,activation="softmax"))


# In[108]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[109]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50,callbacks=[es],verbose=1)


# # CNN-LSTM

# In[110]:


X_train,X_test = X_train.reshape(n_len,1,features),X_test.reshape(n_len_test,1,features)


# In[111]:


Y_train1,Y_test1 = Y_train.reshape(Y_train.shape[0],Y_train.shape[1]),Y_test.reshape(Y_test.shape[0],Y_test.shape[1])


# In[112]:


model = Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(1,features)))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.LSTM(64))
model.add(layers.Flatten())

model.add(layers.Dense(32))
model.add(layers.Dense(6,activation="softmax"))


# In[113]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[114]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50,callbacks=[es],verbose=1)


# # BIRCH

# In[118]:


X_train,X_test = X_train.reshape(n_len,features),X_test.reshape(n_len_test,features)


# In[119]:


model = Birch(n_clusters=6)


# In[120]:


model.fit(X_train,Y_train)


# In[121]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[124]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # DBSCAN

# In[125]:


model = DBSCAN(eps=9.7, min_samples=10, algorithm='auto', metric='euclidean', leaf_size=90, p=2)


# In[126]:


pred_train = model.fit_predict(X_train)
param = model.get_params()


# In[127]:


pred_test = model.fit_predict(X_test )


# # KMeans

# In[128]:


model = KMeans(n_clusters=6)


# In[129]:


model.fit(X_train)


# In[130]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[131]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# #  Mini-Batch K-Means

# In[132]:


model = MiniBatchKMeans(n_clusters=6)


# In[133]:


model.fit(X_train)


# In[134]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[135]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # Mean Shift

# In[136]:


model = MeanShift()


# In[ ]:


model.fit(X_train)


# In[ ]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[ ]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # OPTICS

# In[ ]:


model = OPTICS(min_samples=6)


# In[ ]:


pred_train = model.fit_predict(X_train)
pred_test = model.fit_predict(X_test)


# In[ ]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # Spectral Clustering

# In[ ]:


model = SpectralClustering(n_clusters=6)


# In[ ]:


pred_train = model.fit_predict(X_train)


# In[ ]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))


# # GaussianMixture

# In[ ]:


model = GaussianMixture(n_components=6)


# In[ ]:


model.fit(X_train)


# In[ ]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[ ]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# In[ ]:




