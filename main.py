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

# remove stop words and punctation
for index in xrange(len(article)):
    article[ index ] = article[ index ].translate(string.maketrans("",""), string.punctuation).lower()

article = article[0:300]

# add nodes

previous_ngram = ""

for ngram_index in xrange(len(article)-3):
    if ngram_index == 0:
        ngram = "_".join( article[ngram_index: ngram_index + 3] )
        next_ngram = "_".join( article[ngram_index + 1 : ngram_index + 1 + 3] )

        G.add_node( ngram )
        G.add_node( next_ngram )
    else:
        previous_ngram = ngram
        ngram = next_ngram
        next_ngram = "_".join( article[ngram_index + 1 : ngram_index + 1 + 3] )

        G.add_node( next_ngram )

        G.add_edge( previous_ngram, ngram )
        G.add_edge( ngram, next_ngram )

    #G.add_weighted_edges_from([(1,2,0.125),(1,3,0.75),(2,4,1.2),(3,4,0.375)])


print "drawing..."
nx.draw(G)
print "save plot..."
plt.savefig("path.png")

#print G.nodes()