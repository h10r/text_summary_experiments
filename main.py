# -*- coding: utf-8 -*-

import string
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data

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

def use_lemmatizer():
    lmtzr = WordNetLemmatizer()
    """
    article_tokens[ index ] = lmtzr.lemmatize( article_tokens[ index ] )

    if article_tokens[index] != lmtzr.lemmatize( article_tokens[ index ] ):
        print article_tokens[index], lmtzr.lemmatize( article_tokens[ index ] )
    """

### Load files

with open('common_words.txt', 'r') as f:
    stopwords = f.read().split()

with open('pulman.txt', 'r') as f:
    article = f.read().strip()
    
    article = article[0:10000]

    article_tokens = article.split()


G = nx.Graph()

previous_word = ""

for index in xrange(len(article_tokens)-1):
    # remove punctation
    article_tokens[ index ] = article_tokens[ index ].translate(string.maketrans("",""), string.punctuation)
    article_tokens[ index ] = article_tokens[ index ].lower()

    if index == 0:
        current_word = article_tokens[ index ]
        next_word = article_tokens[ index + 1 ]

        G.add_edge( current_word, next_word )
    else:
        previous_word = current_word
        current_word = next_word
        next_word = article_tokens[ index + 1 ]

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

# tokenize sentences
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sentence_detector.tokenize( article )

sentence_points = [ [i,0] for i in xrange( len( sentences ) ) ]

for sen in xrange(len(sentences)):
    for word_and_rank in ranked_top_100_without_stopwords:
        word, rank = word_and_rank

        if word in sentences[sen]:
            #sentence_points[ sen ][0] += 1
            sentence_points[ sen ][1] += 1

sorted_sentence_points = sorted(sentence_points, key=lambda r: r[1], reverse=True)

N = 5

if len(sorted_sentence_points) > N:
    for i in xrange(N):
        print sentences[ i ]
