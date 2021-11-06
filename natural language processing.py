"""Natural Language Processing."""

# Importing the necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Importing the dataset
dataset = pd.read_csv("reviews_dataset.tsv",
                      delimiter = "\t",
                      quoting = 3)

# Cleaning texts
corpus = []

nltk.download("stopwords")

for i in range(1000):

    review = re.sub("[^a-zA-Z]",
                    " ",
                    dataset["Review"][i])

    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)

    corpus.append(review)

print(corpus)

# Creating Bag of Words

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state = 0)

# Training Naive Bayes model
classifier = GaussianNB()

classifier.fit(X_train, y_train)

# Predicting Test set results
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                      y_test.reshape(len(y_test), 1)),
                     1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)
