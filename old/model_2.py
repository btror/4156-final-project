import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

ax = plt.axes()


def categorize(dataframe):
    """
    Train and test data using a support vector classifier (SVC).

    :param dataframe: A dataframe object containing jeopardy questions.
    :type dataframe: DataFrame
    """

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer="char")
    X = tfidf.fit_transform(dataframe["Question"])
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LinearSVC(C=20, class_weight="balanced", max_iter=20000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("________________________support vector classifier________________________\n")
    print(classification_report(y_test, y_pred))
    print("_________________________________________________________________________\n\n\n")

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


if __name__ == '__main__':
    # read jeopardy data from the csv file
    df = pd.read_csv("../data/jeopardy_data.csv")

    # perform a ML sentiment analysis (support vectorizer classifier)
    categorize(df)

    plt.show()
