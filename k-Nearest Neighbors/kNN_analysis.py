##################################################
# k-Nearest Neighors implementation and comparison
# to sci-kit's kNN implementation. Predicts a song's
# category based off its lyrics. Tokenizes and stems
# each word, then computes cosine similarity between
# one song and all others. Take mode of top k songs
# to predict category.
##################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sqlite3
import random
import math
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
import rrkNN

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

    for row in cursor.fetchall():
        artist, title, time, bpm, year, genre, lyrics, tfidf, newGenre = row
        #tfidf_unloaded = pickle.loads(tfidf)
        training_set_lyrics.append(lyrics)
        training_set_genre.append(newGenre)

    return training_set_lyrics, training_set_genre


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def most_common(lst):
    return max(set(lst), key=lst.count)


def cosine_similarity(song, k):
    cosine_similarities = linear_kernel(X_train, song).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-k:-1]
    words_to_count = Y_train_data[related_docs_indices]
    c = Counter(words_to_count)
    return c.most_common(1)

if __name__ == "__main__":
    #First, read the data from the db and store as numpy arrays
    k = 10
    lyrics, genre = read_final_db()
    lyrics_array = np.asarray(lyrics)
    genre_array = np.asarray(genre)

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

    #Third, fit the data to a naive bayes classifier with scikit
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_data)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train_data)

    X_new_test_data = vectorizer.transform(X_test_data)
    predicted = neigh.predict(X_new_test_data)

    #Fifth, analyze success of scikit
    print(np.mean(predicted == Y_test_data))
    print(metrics.classification_report(Y_test_data, predicted))

    #Sixth, use Ross and Ranney's kNN!
    neighImproved = rrkNN.rrkNN(X_train_data, Y_train_data, X_test_data, Y_test_data, k=5)
    neighImproved.train_corpus()
    predictions = []
    for x in X_new_test_data:
        predictions.append(neighImproved.predict_song(x))
    neighImproved.calculate_statistics(predictions, Y_test_data)

    # for k in range(2, 50):
    #     print k
    #     neighImproved = rrkNN.rrkNN(X_train_data, Y_train_data, X_test_data, Y_test_data, k=k)
    #     neighImproved.train_corpus()
    #     predictions = []
    #     for x in X_new_test_data:
    #         predictions.append(neighImproved.predict_song(x))
    #     neighImproved.calculate_statistics(predictions, Y_test_data)
