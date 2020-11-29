import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

# VISUALIZE THE DATA
stackoverflow_df = pd.read_csv('query.csv')
print(stackoverflow_df.head)
groups = stackoverflow_df.groupby(['tags']).size()
groups.plot.bar()
plt.show()

# TRAIN A SIMPLE BAG OF WORDS MODEL
vectorizer = TfidfVectorizer()
title_features = vectorizer.fit_transform(stackoverflow_df['title'])

# TRAIN AND TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(title_features, stackoverflow_df['tags'],
                                                    test_size=0.33, random_state=42)

# MODEL TRAINING
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

print(predicted)
print(classification_report(y_test, predicted))