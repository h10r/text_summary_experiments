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

#G.add_weighted_edges_from([(1,2,0.125),(1,3,0.75),(2,4,1.2),(3,4,0.375)])

"""
print "drawing..."
nx.draw(G)
print "save plot..."
plt.savefig("path.png")
"""

#print G.nodes()