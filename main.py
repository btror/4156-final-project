import requests
import pandas as pd
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

    # gather html from first 100 pages
    html_data = ""
    for i in range(200):
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
        sentiment = ""
        try:
            title = review.find("a", {"data-hook": "review-title"}).text.strip()
            rating = float(review.find("i", {"data-hook": "review-star-rating"}).text.replace("out of 5 stars", "")
                           .strip())
            description = review.find("span", {"data-hook": "review-body"}).text.replace("Read more", "").strip()
            if rating < 3:
                sentiment = "negative"
            elif rating < 4:
                sentiment = "neutral"
            else:
                sentiment = "positive"
        except AttributeError:
            pass

        # do not include reviews with empty data
        if title != "" and rating != "" and description != "" and "The media could not be loaded." not in description \
                and "a " not in description:
            review_dict = {
                "title": title,
                "rating": rating,
                "description": description,
                "sentiment": sentiment
            }
            reviews.append(review_dict)

    return reviews


def sentiment_analysis(dataframe):
    """
    Train and test data.

    :param dataframe: A dataframe object containing reviews.
    :type dataframe: DataFrame
    """

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer="char")
    X = tfidf.fit_transform(dataframe["description"])
    y = df["sentiment"]  # can change to "rating" to guess 1-5 star instead of negative/positive

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LinearSVC(C=20, class_weight="balanced", max_iter=20000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    y_test = y_test.tolist()
    y_pred = y_pred.tolist()

    custom = "This is a really good game. No problems at all."
    custom = tfidf.transform([custom])
    print("prediction - ", clf.predict(custom))

    plt.plot(y_test, "3", color="red", label="sample rating")
    plt.plot(y_pred, "4", color="blue", label="predicted rating")

    for i in range(len(y_test)):
        plt.vlines(x=i, ymin=y_test[i], ymax=y_pred[i], color="black")

    plt.legend(numpoints=1)
    plt.show()


def save_data(link):
    """
    Save the updated data to the csv file.

    :param link: The link to the product review.
    :type link: str
    """

    # scrape reviews from Amazon.com
    html = get_soup(link)

    # reformat scrapped data (by title, rating, and description)
    customer_reviews = organize_data(html)

    # create a dataframe and write review data to csv file
    dataframe = pd.DataFrame(customer_reviews)
    dataframe.to_csv("data/amazon_review_data.csv", index=False)


if __name__ == '__main__':
    # link to Amazon product reviews
    product_link = "https://www.amazon.com/Xbox-Wireless-Controller-Pulse-Red-Windows-Devices/product-reviews" \
                   "/B0859XT328/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

    # uncomment line below to scrape latest Amazon review data from the product link above
    # save_data(product_link)

    # read amazon review data from the csv file
    df = pd.read_csv("data/amazon_review_data.csv")

    # perform a ML sentiment analysis
    sentiment_analysis(df)
