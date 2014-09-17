# -*- coding: utf-8 -*-

import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

with open('pulman.txt', 'r') as f:
    read_data = f.read()

STOP_WORDS = nltk.corpus.stopwords.words('english') + list( string.punctuation )
tokens = [i.decode('utf-8') for i in nltk.word_tokenize( read_data.lower() ) if i not in STOP_WORDS]

wnl = WordNetLemmatizer()
tokens_lemmatized = [ wnl.lemmatize( i ) for i in tokens]

# model = word2vec.load("data/vectors.bin")

for i in xrange(len(tokens)):
    if tokens[i] != tokens_lemmatized[i]:
        print tokens[i], tokens_lemmatized[i]

"""

tagged = nltk.pos_tag(tokens)

nouns = []
verbs = []

for entity,tag in tagged:
    if "NN" in tag:
        #print entity, tag
        nouns.append( entity )
    if "VB" in tag:
        #print entity, tag
        verbs.append( entity )

print "---"

for elem in nouns:
    print elem

print "---"

for elem in verbs:
    print elem
"""
