import pandas as pd
import numpy as np
import re


def clean_tweets(col):
    col = col.apply(lambda x: re.sub("@[\w]*", "", x)) 
    col = col.apply(lambda x: re.sub("[^a-zA-Z#]", " ", x))
    col = col.apply(lambda x: re.sub(r'\n', ' ', x))
    col = col.apply(lambda x: re.sub(r"http\S+", "", x))
    col = col.apply(lambda x: re.sub(r' +', ' ', x))
    col = col.apply(lambda x: x.strip())
    col = col.apply(lambda x: x.lower())
#     col = col.apply(lambda x: emoji.replace_emoji(x, replace=''))

    return col


def label_encode(col):
    col = col.apply(lambda x: re.sub(r' +', ' ', x))
    col = col.apply(lambda x: x.strip())
    col = col.apply(lambda x: x.lower())
    col = col.replace({'no': 0, 'yes': 1,
                      '2_no_probably_contains_no_false_info': 0,
                      '1_no_definitely_contains_no_false_info': 0,
                      '5_yes_definitely_contains_false_info': 1,
                      '4_yes_probably_contains_false_info': 1,
                      '1_no_definitely_not_harmful': 0,
                      '2_no_probably_not_harmful': 0,
                      '4_yes_probably_harmful': 1,
                      '5_yes_definitely_harmful': 1,
                      'real': 0,
                      'fake': 1})
    
    return col


def check_tweet_len(data, col='tidy_tweet', labels=['q2_label', 'q4_label']):
    data['len_tweet'] = data[col].apply(lambda x: len(x))
    print(data[labels + ['len_tweet']].groupby(labels).describe())
    
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_data(X, y, test_size=0.25, random_state=42):

    if not test_size:
        return X, None, y, None

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test


from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

def word_vector(tokens:list, size:int, keyed_vec:KeyedVectors):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += keyed_vec[word].reshape((1, size))
            count += 1
        except KeyError: 
            # handling the case where the token is not in vocabulary        
            continue
    if count != 0:
        vec /= count
    return vec


def tokens_to_array(tokens:list, size:int, keyed_vec:KeyedVectors):
    array = np.zeros((len(tokens), size))
    for i in range(len(tokens)):
        array[i,:] = word_vector(tokens.iloc[i], size, keyed_vec=keyed_vec)
    return array


def words_embedding(X_train_q4, X_test_q4, model='sentence_transformer'):
    if model=='sentence_transformer':
        sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        X_train, X_test = sentence_transformer.encode(X_train_q4.to_list()), sentence_transformer.encode(X_test_q4.to_list())
    
    if model=='w2v':
        MAX_FEATURE=500
        tokenized_tweet = X_train_q4.apply(lambda x: [i for i in x.split()])
        w2v = Word2Vec(
                sentences=tokenized_tweet,
                size=MAX_FEATURE, 
                window=5, min_count=2, sg=1, 
                hs=0, negative=10, workers=2, 
                seed=34)
        w2v.train(tokenized_tweet, total_examples=len(tokenized_tweet), epochs=20)
        X_train = tokens_to_array(tokenized_tweet, MAX_FEATURE, w2v.wv)
        X_test = tokens_to_array(X_test_q4.apply(lambda x: [i for i in x.split()]), MAX_FEATURE, w2v.wv)
        
    return X_train, X_test

        
def logit_model(X_train, X_test, y_train, y_test, class_weight='balanced'):
    logit = LogisticRegression(class_weight=class_weight, random_state=0)
    logit.fit(X_train, y_train)
    
    print("Test\n")
    print(classification_report(y_test, logit.predict(X_test)))
    print("Train\n")
    print(classification_report(y_train, logit.predict(X_train)))
    

def visualize_2pcs(pcs, y, figsize=(10,10), title=None): 
    colors = ['red', 'blue', 'black', 'orange', 'green', 'gray', 'purple', 'pink']
    fig, ax = plt.subplots(figsize=figsize)
    for i, lab in enumerate(sorted(np.unique(y), reverse=True)):
        plt.scatter(pcs[:,0][y==lab], pcs[:,1][y==lab], c=colors[i], label=lab) 
    
    legend = ax.legend(loc="lower left", title="Classes")
    ax.add_artist(legend)
    plt.title(title)
    plt.show()
    
    
def visualize_3pcs(pcs, y, figsize=(10,10), title=None):
    colors = ['red', 'blue', 'black', 'orange', 'green', 'gray', 'purple', 'pink']
    fig, ax = plt.subplots(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i, lab in enumerate(sorted(np.unique(y), reverse=True)):
        ax.scatter(pcs[:,0][y==lab], pcs[:,1][y==lab], c=colors[i], label=lab) 

    legend = ax.legend(loc="lower left", title="Classes")
    ax.add_artist(legend)
    plt.title(title)
    plt.show()
    