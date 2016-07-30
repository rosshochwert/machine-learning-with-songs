##########################################
# Naive Bayes implementation to predict a
# song's categroy based off of it's lyrics.
# Reads in the lyrics, tokenizes the words,
# and trains the Naive Bayes model.
# Categories: easy litening, rap, country,
# and Post-1980s rock.
##########################################

import nltk
from nltk.stem.porter import PorterStemmer
import sqlite3
import numpy as np
import random
import math

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


def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item.decode("utf-8", "replace")))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def train_corpus():
    for i, x in enumerate(X_train_data):
        category = Y_train_data[i]
        priors[category] += 1
        text = X_train_data[i]
        words = tokenize(text)
        counts = count_words(words)
        for word, count in counts.items():
            # if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
            if word not in vocab:
                vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
            if word not in word_counts[category]:
                word_counts[category][word] = 0.0
            vocab[word] += count
            word_counts[category][word] += count


def calculate_statistics(predictions, actual):
    recall_rap = 0.
    precision_rap = 0.
    tp_rap = 0.
    fp_rap = 0.
    fn_rap = 0.

    for i, x in enumerate(predictions):
        if predictions[i] == actual[i] and actual[i] == "Rap":
            tp_rap += 1
        if predictions[i] == "Rap" and actual[i] != "Rap":
            fp_rap += 1
        if predictions[i] != "Rap" and actual[i] == "Rap":
            fn_rap += 1

    recall_rap = tp_rap / (tp_rap + fn_rap)
    precision_rap = tp_rap / (tp_rap + fp_rap)

    recall_country = 0.
    precision_country = 0.
    tp_country = 0.
    fp_country = 0.
    fn_country = 0.

    for i, x in enumerate(predictions):
        if predictions[i] == actual[i] and actual[i] == "Country":
            tp_country += 1
        if predictions[i] == "Country" and actual[i] != "Country":
            fp_country += 1
        if predictions[i] != "Country" and actual[i] == "Country":
            fn_country += 1

    recall_country = tp_country / (tp_country + fn_country)
    precision_country = tp_country / (tp_country + fp_country)

    recall_rock = 0.
    precision_rock = 0.
    tp_rock = 0.
    fp_rock = 0.
    fn_rock = 0.

    for i, x in enumerate(predictions):
        if predictions[i] == actual[i] and actual[i] == "Post-1980s Rock":
            tp_rock += 1
        if predictions[i] == "Post-1980s Rock" and actual[i] != "Post-1980s Rock":
            fp_rock += 1
        if predictions[i] != "Post-1980s Rock" and actual[i] == "Post-1980s Rock":
            fn_rock += 1

    recall_rock = tp_rock / (tp_rock + fn_rock)
    precision_rock = tp_rock / (tp_rock + fp_rock)

    recall_easy = 0.
    precision_easy = 0.
    tp_easy = 0.
    fp_easy = 0.
    fn_easy = 0.

    for i, x in enumerate(predictions):
        if predictions[i] == actual[i] and actual[i] == "Easy":
            tp_easy += 1
        if predictions[i] == "Easy" and actual[i] != "Easy":
            fp_easy += 1
        if predictions[i] != "Easy" and actual[i] == "Easy":
            fn_easy += 1

    recall_easy = tp_easy / (tp_easy + fn_easy)
    precision_easy = tp_easy / (tp_easy + fp_easy)

    print "Category \t Recall \t Precision"
    print "Rap \t %f \t %f", recall_rap, precision_rap
    print "Country \t %f \t %f", recall_country, precision_country
    print "Post 1980's Rock \t %f \t %f", recall_rock, precision_rock
    print "Easy \t %f \t %f", recall_easy, precision_easy


def predict_song(song):
    words = tokenize(song)
    counts = count_words(words)
    prior_rock = (priors["Post-1980s Rock"] / sum(priors.values()))
    prior_rap = (priors["Rap"] / sum(priors.values()))
    prior_country = (priors["Country"] / sum(priors.values()))
    prior_easy = (priors["Easy"] / sum(priors.values()))

    log_prob_rock = 0.0
    log_prob_country = 0.0
    log_prob_rap = 0.0
    log_prob_easy = 0.0

    for w, cnt in counts.items():
        # skip words that we haven't seen before, or words less than 3 letters long
        if not w in vocab or len(w) <= 3:
            continue
        # calculate the probability that the word occurs at all
        p_word = vocab[w] / sum(vocab.values())
        # for all categories, calculate P(word|category), or the probability a
        # word will appear, given that we know that the document is <category>

        p_w_given_rock = word_counts["Post-1980s Rock"].get(w, 0.0) / sum(word_counts["Post-1980s Rock"].values())
        p_w_given_rap = word_counts["Rap"].get(w, 0.0) / sum(word_counts["Rap"].values())
        p_w_given_easy = word_counts["Easy"].get(w, 0.0) / sum(word_counts["Easy"].values())
        p_w_given_country = word_counts["Country"].get(w, 0.0) / sum(word_counts["Country"].values())

        # add new probability to our running total: log_prob_<category>. if the probability
        # is 0 (i.e. the word never appears for the category), then skip it
        if p_w_given_rock > 0:
            log_prob_rock += math.log(cnt * p_w_given_rock / p_word)
        if p_w_given_rap > 0:
            log_prob_rap += math.log(cnt * p_w_given_rap / p_word)
        if p_w_given_easy > 0:
            log_prob_easy += math.log(cnt * p_w_given_easy / p_word)
        if p_w_given_country > 0:
            log_prob_country += math.log(cnt * p_w_given_country / p_word)

    # print out the reuslts; we need to go from logspace back to "regular" space,
    # so we take the EXP of the log_prob (don't believe me? try this: math.exp(log(10) + log(3)))
    rock_score = (log_prob_rock + math.log(prior_rock))
    country_score = (log_prob_country + math.log(prior_country))
    rap_score = (log_prob_rap + math.log(prior_rap))
    easy_score = (log_prob_easy + math.log(prior_easy))

    winner = max(rock_score, country_score, rap_score, easy_score)

    test_rock = rock_score-winner
    test_country = country_score-winner
    test_rap = rap_score-winner
    test_easy = easy_score-winner

    exp_test_country = math.exp(test_country)
    print exp_test_country

    exp_test_rap = math.exp(test_rap)
    print exp_test_rap

    exp_test_rock = math.exp(test_rock)
    print exp_test_rock

    exp_test_easy = math.exp(test_easy)
    print exp_test_easy

    test_sum = sum(exp_test_rock+exp_test_easy+exp_test_rap+exp_test_country)
    print test_sum

    # take max, subtract from all, then exponentiate. then normalize again. take sum and divide by it. add to one.
    # difference between prediction and base. (ranney knows this)

    if winner == rock_score:
        return "Post-1980s Rock"
    elif winner == country_score:
        return "Country"
    elif winner == rap_score:
        return "Rap"
    elif winner == easy_score:
        return "Easy"


if __name__ == "__main__":
    #Setup some structures to store our data
    vocab = {}
    word_counts = {
        "Easy": {},
        "Rap": {},
        "Country": {},
        "Post-1980s Rock": {}
    }
    priors = {
        "Easy": 0.,
        "Rap": 0.,
        "Country": 0.,
        "Post-1980s Rock": 0.
    }

    lyrics, genre = read_final_db()
    lyrics_array = np.asarray(lyrics)
    genre_array = np.asarray(genre)

    #Second shuffle the data around and split into 1/3 and 2/3 data groups
    indices = np.arange(len(lyrics))
    random.shuffle(indices)

    cutoff = math.floor(len(lyrics)/5)
    test_indices = indices[0:cutoff]
    train_indices = indices[cutoff:-1]

    X_train_data = lyrics_array[train_indices]
    Y_train_data = genre_array[train_indices]

    X_test_data = lyrics_array[test_indices]
    Y_test_data = genre_array[test_indices]

    train_corpus()
    print "Corpus trained"
    predictions = []
    for x in X_test_data:
        predictions.append(predict_song(x))

    correct = 0.
    for i, x in enumerate(predictions):
        if predictions[i] == Y_test_data[i]:
            #correct prediction!
            correct += 1

    accuracy = correct/len(predictions)
    print accuracy
    calculate_statistics(predictions, Y_test_data)
