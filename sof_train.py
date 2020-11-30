import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from sklearn.linear_model import LogisticRegression

''' This script is being used to train models on the top nine languages 
that we have filtered froma  previous script (stackoverflow10.py). First I tried
simple models but currently am in the process of testing Word2Vec '''

questions_tags_combined = pd.read_csv('top_nine_languages.csv')
print(questions_tags_combined)

# STATS

# No. of questions by count
# questions_tags_combined['Tag'].value_counts().plot(kind='bar')
# plt.show()

# No. of words per review
questions_tags_combined['Body_Length'] = questions_tags_combined['Body'].apply(len)
questions_tags_means = questions_tags_combined.groupby('Tag')['Body_Length'].mean()
# questions_tags_means.plot(kind='bar')
# plt.show()

print(questions_tags_combined.dtypes)


# Pre-processing
def pre_processing(column_to_use, output_var):
    questions_tags_combined[column_to_use] = questions_tags_combined[column_to_use].str.lower()

    X_train_, X_test_, y_train_, y_test_ = train_test_split(questions_tags_combined[column_to_use],
                                                            questions_tags_combined[column_to_use],
                                                            stratify=questions_tags_combined[output_var],
                                                            test_size=0.33)
    # Counts
    dv = TfidfVectorizer()
    X_train_ = dv.fit_transform(X_train_)
    X_test_ = dv.transform(X_test_)

    return X_train_, X_test_, y_train_, y_test_


# TRAINING MODELS
# Simplest model - Naive Bayes with title only....
def multinomial_naive_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print(classification_report(y_test, predict))


# X_train, X_test, y_train, y_test = pre_processing('Title', 'Tag')
# multinomial_naive_bayes(X_train, X_test, y_train, y_test)

# Word2Vec embeddings

def tokenize(attribute):
    questions_tags_combined[attribute] = questions_tags_combined[attribute].str.lower()
    tokenized_titles = questions_tags_combined[attribute].apply(word_tokenize)
    #stop_words = set(stopwords.words('english'))
    #tokenized_titles = token for token in tokenized_titles if not token in stop_words

    # model = gensim.models.word2vec()
    '''with open("glove.6B.50d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}'''

    #print(w2v)
    #dim = len(w2v[])
    return tokenized_titles


def apply_word2vec(X, w2v, dim):
    return np.array([
        np.mean([w2v[w] for w in words if w in w2v]
                or [np.zeros(dim)], axis=0)
        for words in X
    ])

# Tokenize all data here
tokenized = tokenize('Title')
questions_tags_combined['tokenized'] = tokenized
print(questions_tags_combined)

# X_train, X_test, y_train, y_test = pre_processing('Title', 'Tag')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
#print(model.vector_size)

# Make list of vectors for each title we have
#tokenized_embedding_list = [model[token] for token in tokenized]
#print(tokenized)

# Allows us to get a vector for each sentence
'''vector_lists = []
for sentence in tokenized:
    vector = [model[word] for word in sentence if word in model.vocab]
    vector_lists.append(vector)
print(vector_lists)'''

'''a = ['Proper', 'way', 'convert', 'html', 'pdf']
vector = model[a]
print(vector)'''

#for
#tokenized_embedding_list = [model[token] for token in tokenized if token in model.vocab]
#print(tokenized_embedding_list.head(5))

# Train Word2Vec
train, test, y_train, y_test = train_test_split(questions_tags_combined.drop(columns='Tag'),
                                                questions_tags_combined['Tag'], test_size=.2,
                                                stratify=questions_tags_combined['Tag'])
tokenized_list = []
for sentence in train['tokenized']:
    tokenized_list.extend(sentence)

print("Number of sentences: {}.".format(len(tokenized_list)))
print("Number of rows: {}.".format(len(train)))

num_features = 300
min_word_count = 3
num_workers = 4
context = 8
downsampling = 1e-3

w2vmodel = gensim.models.Word2Vec(sentences=tokenized_list, sg=1, hs=0,
                                  workers=num_workers, size=num_features, min_count=min_word_count,
                                  window=context, sample=downsampling, negative=5, iter=6)


# Generate sentence vectors from our tokenized sentence of words
def sentence_vectors(model, sentence):
    sentences = [sentence]
    # Collecting all words in the text
    words = np.concatenate(sentences)
    # Collecting words that are known to the model
    model_voc = set(model.wv.vocab.keys())

    sent_vector = np.zeros(model.vector_size, dtype="float32")

    # Use a counter variable for number of words in a text
    nwords = 0
    # Sum up all words vectors that are know to the model
    for word in words:
        if word in model_voc:
            sent_vector += model[word]
            nwords += 1.

    # Now get the average
    if nwords > 0:
        sent_vector /= nwords
    return sent_vector

# Convert this into a consistent set of features for each training and test example
def vectors_to_feats(df, ndim):
    index=[]
    for i in range(ndim):
        df[f'w2v_{i}'] = df['train_vectors'].apply(lambda x: x[i])
        index.append(f'w2v_{i}')
    return df[index]

# Train
train['train_vectors'] = list(map(lambda sen_group: sentence_vectors(w2vmodel, sen_group), train['tokenized']))
X_train = vectors_to_feats(train, 300)
print(X_train.head())

# Test
test['train_vectors'] = list(map(lambda sen_group: sentence_vectors(w2vmodel, sen_group), test['tokenized']))
X_test=vectors_to_feats(test, 300)
print(X_test.head())

# Results
X_train.to_csv('train_sentence_vector_features.csv')
X_test.to_csv('test_sentence_vector_features.csv')

# Logistic Regression Model
clf = LogisticRegression(max_iter=4000)
clf.fit(X_train, y_train)
predict = clf.predict(X_test)

print(classification_report(y_test, predict))
