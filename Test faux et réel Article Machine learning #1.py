#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


donneeText = pd.read_csv('C:\Users\hugoh\Downloads\news.zip')

donneeText.shape
donneeText.head()


# In[4]:


donneeText = pd.read_csv("C:\Users\hugoh\Downloads\news\news.csv")

donneeText.shape
donneeText.head()


# In[5]:


donneeText = pd.read_csv("C:\\Users\\hugoh\\Downloads\\news\\news.csv")

donneeText.shape
donneeText.head()


# In[6]:


labels = donneeText.label
labels.head()


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(donneeText['test'], labels, test_size=0.2, random_state=7)


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(donneeText['test'], labels, test_size=0.2, random_state=7)


# In[10]:


text = donneeText.text
text.head()


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(donneeText['text'], labels, test_size=0.2, random_state=7)


# In[12]:


#la fonction test_size permet de definir le pourcentage de données du dataset que el'on va utiliser pour tester notre model, le random state permet de faire des test tous les x temps, et ensui
#te les variables, x_train sont les données Text d'entrainement, le x_test sont les donnees Text de test, et y sont
#la meme chose mais pour les labels


# In[13]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_donneeText=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.fit_transform(x_test)


# In[14]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.fit_transform(x_test)


# In[15]:


#avec stop words, on ne traite pas les "mots vides" ("is", "the", etc..) donc ceux en anglais ici
#et max df nous permet de si un apparait dans 70% des donnees on l'ignore


# In[16]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[17]:


tfidf_test = tfidf_vectorizer.transform(x_test)


# In[18]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[19]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[20]:


confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])


# In[1]:


print("nombre de données dans le dataset", donneeText.shape)


# In[2]:


donneeText = pd.read_csv("C:\Users\hugoh\Downloads\news\news.csv")

donneeText.shape

print("nombre de données dans le dataset", donneeText.shape)


# In[3]:


donneeText = pd.read_csv("C:\\Users\\hugoh\\Downloads\\news\\news.csv")

donneeText.shape

print("nombre de données dans le dataset", donneeText.shape)


# In[4]:


import pandas as pd

donneeText = pd.read_csv("C:\\Users\\hugoh\\Downloads\\news\\news.csv")

donneeText.shape

print("nombre de données dans le dataset", donneeText.shape)


# In[ ]:




