# import libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# scrape reviews from Amazon.com

# link to Amazon product reviews
product_link = "https://www.amazon.com/Xbox-Wireless-Controller-Pulse-Red-Windows-Devices/product-reviews" \
               "/B0859XT328/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

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
    html_data += requests.get(product_link, headers=HEADERS).text
    product_link = product_link.replace(current_page, next_page)

soup = BeautifulSoup(html_data, "html.parser")

# Collect review titles, ratings, and descriptions.


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

# create a dataframe and write review data to csv file
df = pd.DataFrame(reviews)

# Train and test data using a support vector classifier (SVC).
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer="char")
X = tfidf.fit_transform(df["description"])
y = df["rating"]  # can change to "rating" to guess 1-5 star instead of negative/positive
# y = dataframe["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LinearSVC(C=20, class_weight="balanced", max_iter=20000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("________________________support vector classifier________________________\n")
print(classification_report(y_test, y_pred))
print("_________________________________________________________________________\n\n")

y_test = y_test.tolist()
y_pred = y_pred.tolist()

# statement = "Battlefield 2042 has performance issues"
# s = tfidf.transform([statement])
# print("Prediction: the statement \"" + statement + "\" portrays a " + str(clf.predict(s)[0]) + " sentiment.")

# plot some test data
fig1 = plt.figure()
ax1 = plt.axes()

ax1.plot(y_test[0:50], "3", color="red", label="sample (actual) rating")
ax1.plot(y_pred[0:50], "4", color="blue", label="predicted rating")

for i in range(50):  # len(y_test)
    ax1.vlines(x=i, ymin=y_test[i], ymax=y_pred[i], color="black")

ax1.legend(numpoints=1)
ax1.set_title("sentiment analysis - SVC")
ax1.set_xlabel("50 sample Amazon reviews")
ax1.set_ylabel("review rating")

plt.show()

# Train and test data using a naive bayes.

x = df["description"]
y = df["rating"]
# y = dataframe["sentiment"]

x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

model = MultinomialNB()
model.fit(x, y)

y_pred = model.predict(x_test)

print("_______________________________naive bayes_______________________________\n")
print(classification_report(y_test, y_pred))
print("_________________________________________________________________________\n\n")

# plot
y_test = y_test.tolist()
y_pred = y_pred.tolist()

# statement = "Fortnite is a garbage game"
# s = vec.transform([statement])
# print("Prediction: the statement \"" + statement + "\" portrays a " + str(model.predict(s)[0]) + " sentiment.")

# plot some test data
fig2 = plt.figure()
ax2 = plt.axes()

ax2.plot(y_test[0:50], "3", color="red", label="sample (actual) rating")
ax2.plot(y_pred[0:50], "4", color="blue", label="predicted rating")

for i in range(50):  # len(y_test)
    ax2.vlines(x=i, ymin=y_test[i], ymax=y_pred[i], color="black")

ax2.legend(numpoints=1)
ax2.set_title("sentiment analysis - naive bayes")
ax2.set_xlabel("50 sample Amazon reviews")
ax2.set_ylabel("review rating")

plt.show()
