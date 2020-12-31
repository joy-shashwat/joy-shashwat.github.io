# EDA packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

# Load our data
df = pd.read_csv('names_dataset.csv')

# Checking for Missing Values
df.isnull().isnull().sum()

df_names = df

# Replacing All F and M with 0 and 1 respectively
df_names.sex.replace({'F':0,'M':1},inplace=True)

df_names.sex.unique()

Xfeatures =df_names['name']

# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

from sklearn.model_selection import train_test_split

# Features 
X
# Labels
y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

# Accuracy of our Model
print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")

