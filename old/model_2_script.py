# Import the required libraries.

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# Create a dataframe. Read the Jeopardy questions from the jeopardy_data.csv file
# downloaded from https://www.kaggle.com/tunguz/200000-jeopardy-questions.

df = pd.read_csv("../data/jeopardy_data.csv")

# Use a support vector classifier (SVC) to vectorize the Jeopardy questions.

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer="char")
X = tfidf.fit_transform(df["Question"])
y = df["Category"]

# Train the data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the data.

clf = LinearSVC(C=20, class_weight="balanced", max_iter=20000)
clf.fit(X_train, y_train)

# Display a classification report of the model. Show the accuracy.

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot the model.

fig = plt.figure()
ax = plt.axes()

y_test = y_test.tolist()
y_pred = y_pred.tolist()

ax.plot(y_test[0:50], "3", color="red", label="sample (actual) category")
ax.plot(y_pred[0:50], "4", color="blue", label="predicted category")

for i in range(50):  # len(y_test)
    ax.vlines(x=i, ymin=y_test[i], ymax=y_pred[i], color="black")

ax.legend(numpoints=1)

ax.set_title("classification - SVC")
ax.set_xlabel("50 sample Jeopardy questions")
ax.set_ylabel("category")
