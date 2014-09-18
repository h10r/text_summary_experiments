# -*- coding: utf-8 -*-

import string
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer

### Functions

def most_important(G):
    """ returns a copy of G with
    the most important nodes
    according to the pagerank """ 

    ranking = nx.betweenness_centrality(G).items()
    ranking = sorted(ranking, key=lambda r: r[1], reverse=True)

    return ranking

def pagerank(G):
    """ returns a copy of G with
    the most important nodes
    according to the pagerank """ 

    ranking = nx.betweenness_centrality(G).items()
    #print ranking
    
    r = [x[1] for x in ranking]
    m = sum(r)/len(r) # mean centrality
    t = m*3 # threshold, we keep only the nodes with 3 times the mean

    Gt = G.copy()

    for k, v in ranking:
        if v < t:
            Gt.remove_node(k)
            print v
    return Gt

### Load files

with open('common_words.txt', 'r') as f:
    stopwords = f.read().split()

with open('pulman.txt', 'r') as f:
    article = f.read().split()

G = nx.Graph()

lmtzr = WordNetLemmatizer()

article = article[0:500]

previous_word = ""

for index in xrange(len(article)-1):
    # remove punctation
    article[ index ] = article[ index ].translate(string.maketrans("",""), string.punctuation)
    article[ index ] = article[ index ].lower()

    """
    article[ index ] = lmtzr.lemmatize( article[ index ] )

    if article[index] != lmtzr.lemmatize( article[ index ] ):
        print article[index], lmtzr.lemmatize( article[ index ] )
    """

    if index == 0:
        current_word = article[ index ]
        next_word = article[ index + 1 ]

        G.add_edge( current_word, next_word )
    else:
        previous_word = current_word
        current_word = next_word
        next_word = article[ index + 1 ]

        #print previous_word, current_word, next_word

        G.add_edge( previous_word, current_word )
        G.add_edge( current_word, next_word )
        
        G.add_edge( previous_word, next_word )

ranked = most_important(G)

ranked_top_100_without_stopwords = []

count = 0
index = 0
max_index = len( ranked )

while count <= 42:
    word,ranking = ranked[ index ]

    if not word in stopwords:
        ranked_top_100_without_stopwords.append( (word, ranking) )
        count += 1

    index += 1

    if index >= max_index:
        break

for r in ranked_top_100_without_stopwords:
    print r

# @TODO: Add word2vec weights

# @TODO: Select and print most important sentences


