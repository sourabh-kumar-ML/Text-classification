#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
from string import punctuation

import nltk
nltk.download('stopwords')


# # Data Preprocessing

# In[19]:


data_train = pd.read_csv("train.tsv", sep="\t",header=None)
data_train.index = [i for i in range(0,len(data_train))]


# In[20]:


data_test = pd.read_csv("test.tsv", sep="\t",header=None)
data_test.index = [i for i in range(0,len(data_test))]


# In[21]:


data_train.head()


# In[22]:


data_test.head()


# In[23]:


data_train.shape


# In[24]:


data_train = data_train[[1,2]]
data_test = data_test[[1,2]]


# In[25]:


data_train.columns = ["label","text"]
data_test.columns = ["label","text"]


# In[26]:


data_train.head()


# In[27]:



def clean_doc(doc):
    # split into tokens by white space
    rtr = []
    for tokens in doc:
        tokens = tokens.split()
        # remove punctuation from each token
        tokens = [word.lower() for word in tokens]
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        if len(tokens) > 0:
            rtr.append(" ".join(tokens))
    return rtr    


# In[28]:


data_train.text = clean_doc(data_train.text)
data_test.text = clean_doc(data_test.text)


# In[29]:


data_train.head()


# In[17]:


data_train.to_csv("train_extracted.csv",index=None)


# In[18]:


data_test.to_csv("test_extracted.csv",index=None)


# In[ ]:




