# -*- coding: utf-8 -*-

import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

with open('pulman.txt', 'r') as f:
    article = f.read().split()

STOP_WORDS = nltk.corpus.stopwords.words('english') + list( string.punctuation )

#tokens = [i.decode('utf-8') for i in nltk.word_tokenize( article.lower() ) if i not in STOP_WORDS]

G = nx.Graph()

#article = article[0:300]

previous_word = ""

for index in xrange(len(article)-1):
    
    # remove stop words and punctation
    article[ index ] = article[ index ].translate(string.maketrans("",""), string.punctuation).lower()

    if index == 0:
        current_word = article[ index ]
        next_word = article[ index + 1 ]

        G.add_edge( current_word, next_word )
    else:
        previous_word = current_word
        current_word = next_word
        next_word = article[ index + 1 ]

        print previous_word, current_word, next_word

        G.add_edge( previous_word, current_word )
        G.add_edge( current_word, next_word )
        
        G.add_edge( previous_word, next_word )

def most_important(G):
    """ returns a copy of G with
    the most important nodes
    according to the pagerank """ 

    ranking = nx.betweenness_centrality(G).items()
    print ranking
    r = [x[1] for x in ranking]
    m = sum(r)/len(r) # mean centrality
    t = m*3 # threshold, we keep only the nodes with 3 times the mean
    Gt = G.copy()
    for k, v in ranking:
        if v < t:
            Gt.remove_node(k)
    return Gt

Gt = most_important(G) # trimming

from pylab import show

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.2,node_size=8)
nx.draw_networkx_edges(G,pos,alpha=0.1)

nx.draw_networkx_nodes(Gt,pos,node_color='r',alpha=0.4,node_size=254)

nx.draw_networkx_labels(Gt,pos,font_size=12,font_color='b')

show()

