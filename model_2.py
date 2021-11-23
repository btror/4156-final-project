import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


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

    # custom = "I like equations"
    # custom = tfidf.transform([custom])
    # print("prediction - ", clf.predict(custom))

    plt.plot(y_test, "3", color="red", label="sample rating")
    plt.plot(y_pred, "4", color="blue", label="predicted rating")

    for i in range(len(y_test)):
        plt.vlines(x=i, ymin=y_test[i], ymax=y_pred[i], color="black")

    plt.legend(numpoints=1)


if __name__ == '__main__':

    # read jeopardy data from the csv file
    df = pd.read_csv("data/jeopardy_data.csv")

    # History, Science, Literature
    df.to_csv("data/jeopardy_data.csv", index=False)

    # perform a ML sentiment analysis (support vectorizer classifier)
    categorize(df)

    plt.show()
