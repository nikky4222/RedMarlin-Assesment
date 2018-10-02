# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:27:21 2018

@author: nikit
"""
'''
Copyright <2018> <Nikitha Kona>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
'''
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
#Dataset location: https://s3-us-west-2.amazonaws.com/rm-exercise/rm_topic_modeling_data.zip

path=r'C:\Users\nikit\OneDrive\Documents\RedMarlin\topic_modeling_data.json'
data = []

with open(path) as f:
    for line in f:
        js = json.loads(line)
        data.append(json.loads(line))
        
g=pd.DataFrame(data)
g['_id']=g['_id'].apply(lambda x:str(x))
g['text']=g['text'].apply(lambda x:str(x)) 

text_data = [] #converting the data into a dataframe

spacy.load('en')
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:    #preprocessing the data by performing tokenization
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
        return word            #preprocessing the data by performing lemmatization
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
    corpus = [dictionary.doc2bow(text) for text in text_data]  #Applying LDA to get the important topics
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    NUM_TOPICS = 1
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=5)
    fin.append(topics)
        
fin2=pd.DataFrame(fin)
j_fin=pd.concat([g['_id'],fin2],axis=1)
j_fin.columns=['id','topics']
for i,jj in enumerate(j_fin['topics']):
    n=list(jj)
    n1=n[1:]                                    #removing the weights from the final result 
    m=''.join(n1)
    mn=m.split("*")
    mn=mn[1:]
    
    j_fin['topics'].iloc[i]=[f.split("+")[0] for f in mn ] #j_fin has the final result with{id,topics}
    a=j_fin['topics'].iloc[i]
    aa=''.join(a)
    a2=aa.replace('"', '')
    #print(a2)
    j_fin['topics'].iloc[i]=a2.split(" ")

#with open('temp.json', 'w') as f:
    #f.write(j_fin.to_json(orient='records', lines=True))
md=pd.DataFrame(j_fin.topics.tolist())
#md = pd.to_numeric(md, errors='coerce')
md1=j_fin.set_index('id').T.to_dict('list')
        
f=open("finall.json","w")
for k,v in md1.items():
   s=str(k)+"   "+str(v)+"\n"
   b=f.write(s)