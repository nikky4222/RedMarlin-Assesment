# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:27:21 2018

@author: nikit
"""
import json
import pandas as pd
from gensim import corpora
import spacy
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import gensim

path=r'C:\Users\nikit\OneDrive\Documents\RedMarlin\topic_modeling_data.json'
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))
g=pd.DataFrame(data)
g['_id']=g['_id'].apply(lambda x:str(x))
g['text']=g['text'].apply(lambda x:str(x))
text_data = []

spacy.load('en')
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

fin=[]
for line in g['text']:
    text_data=[]
    tokens = prepare_text_for_lda(line)
    text_data.append(tokens)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    NUM_TOPICS = 1
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=5)
    #for topic in topics:
    fin.append(topics)
        
fin2=pd.DataFrame(fin)
j_fin=pd.concat([g['_id'],fin2],axis=1)
j_fin.columns=['id','topics']
for i,jj in enumerate(j_fin['topics']):
    n=list(jj)
    n1=n[1:]
    m=''.join(n1)
    mn=m.split("*")
    mn=mn[1:]
    
    j_fin['topics'].iloc[i]=[f.split("+")[0] for f in mn ]
    