#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.python.keras.layers as layers
from tensorflow.python.keras.models import Sequential
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,OPTICS,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
tf.config.experimental_run_functions_eagerly(True)


# In[2]:


data_train = pd.read_csv("train_extracted.csv")
data_test = pd.read_csv("test_extracted.csv")


# In[3]:


oe_enc = OneHotEncoder()
tfidf_vectorizer = TfidfVectorizer()


# In[4]:


train_labels = data_train.label
train_text = data_train.text

test_labels = data_test.label
test_text = data_test.text


# In[5]:


X_train = tfidf_vectorizer.fit_transform(train_text).toarray()
X_test = tfidf_vectorizer.transform(test_text).toarray()


# In[6]:


Y_train = oe_enc.fit_transform(np.array(train_labels).reshape(-1,1)).toarray()
Y_test = oe_enc.fit_transform(np.array(test_labels).reshape(-1,1)).toarray()


# In[7]:


true_label_train = np.argmax(Y_train,axis=1)
true_label_test = np.argmax(Y_test,axis=1)


# In[8]:


X_train.shape,X_test.shape


# In[9]:


n_len,features = X_train.shape
n_len_test,features = X_test.shape


# In[10]:


optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)


# # RNN

# In[ ]:


model = Sequential(name="RNN")
model.add(layers.Embedding(input_dim=X_train.shape[1], output_dim=64))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(6,activation="softmax"))


# In[ ]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


model.fit(X_train,Y_train,validation_data = (X_test,Y_test),callbacks=[es],epochs=50)


# # LSTM

# In[ ]:


X_train,X_test = X_train.reshape(n_len,features,1),X_test.reshape(n_len_test,features,1)


# In[ ]:


model = Sequential(name="LSTM")
model.add(layers.LSTM(128,input_shape=(X_train.shape[1],1)))
model.add(layers.Dense(6,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer = optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),callbacks=[es],epochs=50,verbose=1)


# # Bi-LSTM

# In[ ]:


model=Sequential(name="Bi-LSTM")
model.add(layers.Bidirectional(LSTM(64,input_shape=(X_train.shape[1],1),activation="relu")))
model.add(layers.Dense(6,activation="softmax"))


# In[ ]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


model.fit(X_train,Y_train,validation_data = (X_test,Y_test),callbacks=[es],epochs=50)


# # CNN

# In[ ]:


model = Sequential(name = "CNN")
model.add(layers.Conv1D(filters=32,kernel_size=4,input_shape=(X_train.shape[1],1),activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(filters=64,kernel_size=6,activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Dense(6,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50,callbacks=[es],verbose=1)


# # CNN-LSTM

# In[52]:


X_train,X_test = X_train.reshape(n_len,1,features),X_test.reshape(n_len_test,1,features)


# In[54]:


Y_train1,Y_test1 = Y_train.reshape(Y_train.shape[0],Y_train.shape[1]),Y_test.reshape(Y_test.shape[0],Y_test.shape[1])


# In[55]:


model = Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(1,features)))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.LSTM(64))
model.add(layers.Flatten())

model.add(layers.Dense(32))
model.add(layers.Dense(6,activation="softmax"))


# In[56]:


model.summary()


# In[57]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])


# In[58]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50,callbacks=[es],verbose=1)


# # BIRCH

# In[59]:


X_train,X_test = X_train.reshape(n_len,features),X_test.reshape(n_len_test,features)


# In[60]:


model = Birch(n_clusters=6)


# In[61]:


model.fit(X_train,Y_train)


# In[62]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[76]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # DBSCAN

# In[45]:


model = DBSCAN(eps=9.7, min_samples=10, algorithm='auto', metric='euclidean', leaf_size=90, p=2)


# In[48]:


pred_train = model.fit_predict(X_train)
param = model.get_params()


# In[46]:


pred_test = model.fit_predict(X_test )


# # K-Means

# In[51]:


model = KMeans(n_clusters=6)


# In[52]:


model.fit(X_train)


# In[53]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[54]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# #  Mini-Batch K-Means

# In[55]:


model = MiniBatchKMeans(n_clusters=6)


# In[56]:


model.fit(X_train)


# In[57]:


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


# In[58]:


print("Train Accuracy :: ",accuracy_score(true_label_train,pred_train))
print("Test Accuracy :: ",accuracy_score(true_label_test,pred_test))


# # Mean Shift

# In[59]:


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

