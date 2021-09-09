# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 07:51:32 2021

@author: Rodrigo

For the below example, we have used the data found on:
https://www.kaggle.com/fireball684/hackerearthericsson
    
The objective is to find topics within the data set which is comprised of employee reviews
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF
import numpy as np


employee_reviews = pd.read_csv("employee_reviews.csv")
#Check columns and types
employee_reviews.dtypes
#select strings
str_cols = employee_reviews.select_dtypes(['object']).columns
#Trim all columns
employee_reviews[str_cols] = employee_reviews[str_cols].apply(lambda x: x.str.strip())


#Splitting two samples from the same company (startup_1) between former and current employees
employee_review = employee_reviews[(employee_reviews["Place"] == "startup_1")]

#Function for lemmatizing the title and removing punctuation. 
def cleanText(text):
    lemmatizer = WordNetLemmatizer() 
    #Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split(" ")
    clean_text = []
    for word in tokens:
        clean_text.append(lemmatizer.lemmatize(word))
    return " ".join(clean_text)

def applyCleanText(reviews,col="positives",output="comment"):
    for index, row in reviews.iterrows():
        comment = row[col]
        comment= cleanText(comment)
        reviews.at[index, output] = comment
    return reviews

employee_review = applyCleanText(employee_review)


#create tfidf vector 
tfidf_vec_current = TfidfVectorizer(max_features=500,stop_words="english")
tfidf_current = tfidf_vec_current.fit_transform(employee_review["comment"])
#number of components is unknown, you have to optimize by trial and error unless you know the data. Here we would like to retrieve the top 5 topics
nmf=NMF(n_components=5, random_state=1,beta_loss="kullback-leibler",solver="mu",alpha=.1,l1_ratio=.5,max_iter=1000).fit(tfidf_current)

tfidf_transform_current = nmf.transform(tfidf_current)
#


#Assign cluster (topics)
cluster = np.argmax(tfidf_transform_current, axis=1)+1
employee_review["cluster"] = cluster


#number of reviews per topic
topics, count = np.unique(cluster,return_counts=True)
np.asarray((topics, count))

#number of words
n=10
feature_names = tfidf_vec_current.get_feature_names()

#Extract them top 10 words within each topic
df_data = []
for index,topic in enumerate(nmf.components_):
    top_words_idx = topic.argsort()[:-n-1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    tfidf= topic[top_words_idx].tolist()
    df_data.append([index,top_words,tfidf])

    
index = sum([[item[0]] *10 for item in df_data],[])
words = sum([item[1] for item in df_data],[])
tfidf = sum([item[2] for item in df_data],[])

#print topic, topwords, tfidf
top_words = pd.DataFrame({"topic":index,"words":words,"tfidf":tfidf})
color=["red","blue","green","black","grey"]
#Plot words
plt_color = sum([[item] *10 for item in color],[])

top_words.plot.bar(x="words",y="tfidf",color = plt_color,legend=False,figsize=(10,10))