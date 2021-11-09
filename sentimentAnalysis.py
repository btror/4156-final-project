import nltk
import requests
import pandas as pd
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC


def get_soup(url):
    """
    Get the html data for the reviews on the first 20 pages.

    :param url: The url of the first page of reviews.
    :type url: str

    :return: HTML data.
    """

    HEADERS = ({'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/90.0.4430.212 Safari/537.36',
                'Accept-Language': 'en-US, en;q=0.5'})

    # gather html from first 20 pages
    html_data = ""
    for i in range(50):
        # adjust the link to loop through each page
        current_page = str("=" + str(int((i + 1))))
        next_page = str("=" + str(int((i + 2))))
        html_data += requests.get(url, headers=HEADERS).text
        url = url.replace(current_page, next_page)

    soup = BeautifulSoup(html_data, "html.parser")
    return soup


def organize_data(soup):
    """
    Collect review titles, ratings, and descriptions.

    :param soup: The HTML data.
    :type soup: bs4

    :return: A list of reviews.
    """

    html_data = soup.find_all("div", {"data-hook": "review"})

    # store each review in a list
    reviews = []
    for review in html_data:

        # collect the review title, rating, and description
        title = ""
        rating = ""
        description = ""
        try:
            title = review.find("a", {"data-hook": "review-title"}).text.strip()
            rating = float(review.find("i", {"data-hook": "review-star-rating"}).text.replace("out of 5 stars", "")
                           .strip())
            description = review.find("span", {"data-hook": "review-body"}).text.replace("Read more", "").strip()
        except AttributeError:
            pass

        # do not include reviews with empty data
        if title != "" and rating != "" and description != "" and "The media could not be loaded." not in description:
            review_dict = {
                "title": title,
                "rating": rating,
                "description": description,
            }
            reviews.append(review_dict)

    return reviews


def sentiment_analysis(dataframe):
    # TODO: create a sentiment analysis ML algorithm. Determine if a positive or negative emotion is portrayed (Naive
    #  Bayes) check this tutorial out: https://www.twilio.com/blog/2017/12/sentiment-analysis-scikit-learn.html
    data = dataframe["description"]
    data_ratings = dataframe["rating"]

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer="char")
    X = tfidf.fit_transform(dataframe["description"])
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LinearSVC(C=20, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))




    # vectorizer = CountVectorizer(
    #     analyzer="word",
    #     lowercase=False,
    # )
    # features = vectorizer.fit_transform(
    #     data
    # )
    # features_nd = features.toarray()
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     features_nd,
    #     data_ratings,
    #     train_size=.1,
    #     random_state=0)
    #
    # log_model = LogisticRegression()
    # log_model = log_model.fit(X=X_train, y=y_train)
    # y_pred = log_model.predict(X_test)
    #
    # print(accuracy_score(y_test, y_pred))
    # print(y_pred)
    #
    # plt.figure(figsize=(8, 5))
    # plt.title("Multiple Features", loc="left")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    #
    # plt.plot([0, 20], [0, 20], color="red")
    #
    # plt.scatter(y_pred, y_test, color="blue", marker="+")
    #
    # plt.xlim([0, 20])
    # plt.ylim([0, 20])
    #
    # plt.show()


if __name__ == '__main__':
    # link for ZOTAC GAMING GeForce GTX 1650 graphics card
    link = "https://www.amazon.com/Xbox-Wireless-Controller-Pulse-Red-Windows-Devices/product-reviews/B0859XT328/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

    # scrape reviews from Amazon.com
    html = get_soup(link)

    # reformat scrapped data (by title, rating, and description)
    customer_reviews = organize_data(html)

    # create a dataframe and write review data to csv file
    df = pd.DataFrame(customer_reviews)
    df.to_csv("data/amazon_review_data.csv", index=False)

    # gather 7 reviews from every rating
    # size = df[df.rating == 2.0].shape[0]
    # one_star = df[df.rating == 1.0].head(size)
    # two_star = df[df.rating == 2.0].head(size)
    # three_star = df[df.rating == 3.0].head(size)
    # four_star = df[df.rating == 4.0].head(size)
    # five_star = df[df.rating == 5.0].head(size)

    # df = pd.concat([one_star, two_star, three_star, four_star, five_star])

    # perform a ML sentiment analysis
    sentiment_analysis(df)