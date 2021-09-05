# -*- coding: utf-8 -*-
"""
Created on Sun July  5 09:03:06 2021

@author: Rodrigo

The purpose of this document is to analyze the posts in the subreddit wallstreetbets from January 2019 to February 2021. The dataset is available at https://www.kaggle.com/unanimad/reddit-rwallstreetbets. The dataset contains several variables, but for the purpose of this document we are taking into account the following ones:

* title: the title of the post.
* created_utc: date of the post publication.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.stem import WordNetLemmatizer
import emoji
import functools
import operator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

wstbets = pd.read_csv("py_wallstreetbets_posts.csv", usecols=['title', 'created_utc'])

wstbets.head()

#Convert unix timestamp to datetime

wstbets['date'] = pd.to_datetime(wstbets["created_utc"], unit='s')

wstbets_recent =  wstbets[wstbets["date"] >= '2019']



"""
We start by taking a sample of 50,000 observations. Next, we convert the vector containing the title of the post to a corpus, which is just a collection of text documents.
"""

wstbets_sample = wstbets_recent.sample(n=50000, random_state=1)

#Function to split emojis that are not separated by whitespaces
def splitEmoji(sentence):
    emojis = emoji.get_emoji_regexp().split(sentence)
    split_ws = [substr.split() for substr in emojis]
    return functools.reduce(operator.concat, split_ws)

#Function for lemmatizing the title. Keeping numbers as price predictions are important. Also keeping emojis as they are the main way of communicating 
def cleanTitle(title):
    lemmatizer = WordNetLemmatizer() 
    #Remove punctuations
    #title = re.sub(r'[^\w\s]','',title)
    title = title.translate(str.maketrans('', '', string.punctuation))
    #use splitEmoji
    tokens = splitEmoji(title)
    clean_title = []
    for word in tokens:
        clean_title.append(lemmatizer.lemmatize(word))
    return " ".join(clean_title)

#This may take many seconds
for index, row in wstbets_sample.iterrows():
    title = row["title"]
    title = cleanTitle(title)
    wstbets_sample.at[index, "new_title"] = title


count_vec = CountVectorizer(max_features=500, token_pattern=r'[^\s]+', stop_words="english")


"""
We then convert the corpus to a Term Document Matrix, which is a representation of our text documents in a matrix of numbers. The below table shows a subset of the Term Document Matrix. 
"""
cps = count_vec.fit_transform(wstbets_sample["new_title"])

df = pd.DataFrame(cps.A, columns=count_vec.get_feature_names())

#Plot most common terms
plt.rcParams["font.family"] = "Segoe UI Emoji"
term_frequency = df.sum()
term_frequency_top = term_frequency[term_frequency > 1000].sort_values()
term_frequency_top.plot.bar(title="Most frequent terms")
plt.show();

"""
We finish our review by plotting a network to highlight which words are more often linked to GME. We could also have used a clustering algorithm, but we have opted to a graph network for visualization purposes. 
"""

#Adjaceny matrix for GME

gme = df[df["gme"] > 0]
#Convert to term-term adj matrix. If term appears more than once in the document, it should be counted only once
gme[gme >= 1] = 1
gme_x = np.dot(np.transpose(gme),gme)
gme_adj_matrix = pd.DataFrame(gme_x, gme.columns,gme.columns)
gme_index = gme_adj_matrix[gme_adj_matrix["gme"] >200].index
gme_adj_matrix = gme_adj_matrix[gme_adj_matrix["gme"] >200][gme_index]

#Plot network graph

gme_graph = nx.from_pandas_adjacency(gme_adj_matrix)
nx.draw_networkx(gme_graph, font_family="Segoe UI Emoji",node_size=np.diag(gme_adj_matrix)/1.5, node_color='lightblue',edge_color='lightgray')





