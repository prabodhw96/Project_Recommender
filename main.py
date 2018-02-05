import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class Parts2Project():
    def __init__(self):
        # Read data.csv into pandas dataframe
        data = pd.read_csv('data.csv',  delimiter=',')
        # Convert dataframe to list
        self.parts = data['parts']
        # Select and assign the url column to projects list
        project = data['url']
        # Convert names of parts to matrix of token counts
        vectorizer = CountVectorizer()
        self.X = vectorizer.fit_transform(self.parts).toarray()
        # Convert the project dataframe into a numpy array
        self.y = np.array(project)

    def model(self, X_train, y_train, parts_tk):
        # Implement the Multinomial Naive Bayes algorithm for classification
        clf = MultinomialNB()
        # Fit features and labels into classifier
        clf.fit(X_train, y_train)
        # Convert feature to predict into a matrix of token count
        vectorizer = CountVectorizer()
        vectorizer.fit(self.parts)
        parts_tk = vectorizer.transform([parts_tk]).toarray()
        # Predict
        probs = clf.predict_proba(parts_tk)
        recommendations = sorted(zip(clf.classes_, probs[0]), key=lambda x:x[1])[-3:]
        return recommendations


if __name__ == '__main__':
    # Instance of class Parts2Project
    pp = Parts2Project()
    # Ask user for input of parts
    parts_taken = input("Parts taken from the inventory:")
    # Prediction
    prediction = pp.model(pp.X, pp.y, parts_taken)
    print("\n\nRecommended projects to explore:")
    for i in prediction:
        print(i[0])
