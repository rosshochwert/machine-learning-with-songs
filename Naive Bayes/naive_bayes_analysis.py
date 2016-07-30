##########################################
# This script compates the Ross-Ranney
# Naive Bayes implementation with the Sci-kit
# implementation. It also looks at our
# most "confident" errors: songs we were
# convinced were one category, but ended up
# being another category.
#######################################


import sqlite3
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn import metrics
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import rrNB

stemmer = PorterStemmer()


def read_final_db():
    """Read Final DB"""

    sqlite_file = 'final-processed.db'
    connector = sqlite3.connect(sqlite_file)
    connector.row_factory = sqlite3.Row

    #to accomodate the pickle data, need to set this to string so sql doesn't try to decode to utf-8.
    connector.text_factory = str

    cursor = connector.cursor()
    cursor.execute("select * from songs where newGenre='Easy' OR newGenre='Rap' OR newGenre='Country' or newGenre='Post-1980s Rock'")

    training_set_lyrics = list()
    training_set_genre = list()
    training_set_artist = list()
    training_set_title = list()

    for row in cursor.fetchall():
        artist, title, time, bpm, year, genre, lyrics, tfidf, newGenre = row
        #tfidf_unloaded = pickle.loads(tfidf)
        training_set_lyrics.append(lyrics)
        training_set_genre.append(newGenre)
        training_set_artist.append(artist)
        training_set_title.append(title)

    return training_set_lyrics, training_set_genre, training_set_artist, training_set_title


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def test_sckikit_naive_bayes():

    #First, read the data from the db and store as numpy arrays
    lyrics, genre, artist, title = read_final_db()
    lyrics_array = np.asarray(lyrics)
    genre_array = np.asarray(genre)

    artist_array = np.asarray(artist)
    title_array = np.asarray(title)

    #Second, shuffle the data around and split into 1/4 and 3/4 data groups
    indices = np.arange(len(lyrics))
    random.shuffle(indices)

    cutoff = math.floor(len(lyrics)/4)
    test_indices = indices[0:cutoff]
    train_indices = indices[cutoff:-1]

    X_train_data = lyrics_array[train_indices]
    Y_train_data = genre_array[train_indices]

    X_test_data = lyrics_array[test_indices]
    Y_test_data = genre_array[test_indices]

    title_test_data = title_array[test_indices]
    artist_test_data = artist_array[test_indices]

    #Third, fit the data to a naive bayes classifier with scikit
    # vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
    # X_train = vectorizer.fit_transform(X_train_data)
    # clf = MultinomialNB()
    # clf.fit(X_train, Y_train_data)

    # #Fourth, predict
    # X_new_test_data = vectorizer.transform(X_test_data)
    # predicted = clf.predict(X_new_test_data)

    # #Fifth, analyze success of scikit
    # print(np.mean(predicted == Y_test_data))
    # print(metrics.classification_report(Y_test_data, predicted))

    #Sixth, use ross/ranney naive bayes!
    clfImproved = rrNB.rrNB(X_train_data, Y_train_data, X_test_data, Y_test_data)
    clfImproved.train_corpus()
    predictions = []
    final_scores = []
    for x in X_test_data:
        category, final_scores_dict = clfImproved.predict_song(x)
        predictions.append(category)
        final_scores.append(final_scores_dict)
    clfImproved.calculate_statistics(predictions, Y_test_data)

    rap_errors = clfImproved.rap_error(final_scores, predictions, Y_test_data)

    print "Rap song mistaken for Rock:"
    print title_test_data[rap_errors["rap_rock"]]
    print artist_test_data[rap_errors["rap_rock"]]

    print "Rap song mistaken for Easy:"
    print title_test_data[rap_errors["rap_easy"]]
    print artist_test_data[rap_errors["rap_easy"]]

    print "Rap song mistaken for Country:"
    print title_test_data[rap_errors["rap_country"]]
    print artist_test_data[rap_errors["rap_country"]]

    rock_errors = clfImproved.rock_error(final_scores, predictions, Y_test_data)

    print "---------------------------"
    print "Rock song mistaken for Rap:"
    print title_test_data[rock_errors["rock_rap"]]
    print artist_test_data[rock_errors["rock_rap"]]

    print "Rock song mistaken for Country:"
    print title_test_data[rock_errors["rock_country"]]
    print artist_test_data[rock_errors["rock_country"]]

    print "Rock song mistaken for Easy:"
    print title_test_data[rock_errors["rock_easy"]]
    print artist_test_data[rock_errors["rock_easy"]]

    country_errors = clfImproved.country_error(final_scores, predictions, Y_test_data)

    print "---------------------------"
    print "Country song mistaken for Rock:"
    print title_test_data[country_errors["country_rock"]]
    print artist_test_data[country_errors["country_rock"]]

    print "Country song mistaken for Easy:"
    print title_test_data[country_errors["country_easy"]]
    print artist_test_data[country_errors["country_easy"]]

    print "Country song mistaken for Rap:"
    print title_test_data[country_errors["country_rap"]]
    print artist_test_data[country_errors["country_rap"]]

    easy_errors = clfImproved.easy_error(final_scores, predictions, Y_test_data)

    print "---------------------------"
    print "Easy song mistaken for Rock:"
    print title_test_data[easy_errors["easy_rock"]]
    print artist_test_data[easy_errors["easy_rock"]]

    print "Easy song mistaken for Country:"
    print title_test_data[easy_errors["easy_country"]]
    print artist_test_data[easy_errors["easy_country"]]

    print "Easy song mistaken for Rap:"
    print title_test_data[easy_errors["easy_rap"]]
    print artist_test_data[easy_errors["easy_rap"]]

if __name__ == "__main__":
    test_sckikit_naive_bayes()
