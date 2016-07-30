##########################################
# Ross-Ranney Naive Bayes object.
# Pretty dope class if you ask me. 
#######################################

import nltk
from nltk.stem.porter import PorterStemmer
import math
from tabulate import tabulate
from collections import Counter


class rrNB(object):

    def __init__(self, X_train_data, Y_train_data, X_test_data, Y_test_data):
        self.X_train_data = X_train_data
        self.Y_train_data = Y_train_data

        self.X_test_data = X_test_data
        self.Y_test_data = Y_test_data

        self.vocab = {}
        self.word_counts = {
            "Easy": {},
            "Rap": {},
            "Country": {},
            "Post-1980s Rock": {}
        }
        self.priors = {
            "Easy": 0.,
            "Rap": 0.,
            "Country": 0.,
            "Post-1980s Rock": 0.
        }

        self.stemmer = PorterStemmer()

    def count_words(self, words):
        wc = {}
        for word in words:
            wc[word] = wc.get(word, 0.0) + 1.0
        return wc

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems

    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item.decode("utf-8", "replace")))
        return stemmed

    def train_corpus(self):
        for i, x in enumerate(self.X_train_data):
            category = self.Y_train_data[i]
            self.priors[category] += 1
            text = self.X_train_data[i]
            words = self.tokenize(text)
            counts = self.count_words(words)
            for word, count in counts.items():
                # if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
                if word not in self.vocab:
                    self.vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
                if word not in self.word_counts[category]:
                    self.word_counts[category][word] = 0.0
                self.vocab[word] += count
                self.word_counts[category][word] += count

    def predict_song(self, song):
        words = self.tokenize(song)
        counts = self.count_words(words)
        prior_rock = (self.priors["Post-1980s Rock"] / sum(self.priors.values()))
        prior_rap = (self.priors["Rap"] / sum(self.priors.values()))
        prior_country = (self.priors["Country"] / sum(self.priors.values()))
        prior_easy = (self.priors["Easy"] / sum(self.priors.values()))

        log_prob_rock = 0.0
        log_prob_country = 0.0
        log_prob_rap = 0.0
        log_prob_easy = 0.0

        for w, cnt in counts.items():
            # skip words that we haven't seen before, or words less than 3 letters long
            if not w in self.vocab or len(w) <= 3:
                continue
            # calculate the probability that the word occurs at all
            p_word = self.vocab[w] / sum(self.vocab.values())
            # for all categories, calculate P(word|category), or the probability a
            # word will appear, given that we know that the document is <category>

            p_w_given_rock = self.word_counts["Post-1980s Rock"].get(w, 0.0) / sum(self.word_counts["Post-1980s Rock"].values())
            p_w_given_rap = self.word_counts["Rap"].get(w, 0.0) / sum(self.word_counts["Rap"].values())
            p_w_given_easy = self.word_counts["Easy"].get(w, 0.0) / sum(self.word_counts["Easy"].values())
            p_w_given_country = self.word_counts["Country"].get(w, 0.0) / sum(self.word_counts["Country"].values())

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

        exp_test_rap = math.exp(test_rap)

        exp_test_rock = math.exp(test_rock)

        exp_test_easy = math.exp(test_easy)

        test_sum = exp_test_rock+exp_test_easy+exp_test_rap+exp_test_country

        exp_final_country = exp_test_country/test_sum
        exp_final_rock = exp_test_rock/test_sum
        exp_final_rap = exp_test_rap/test_sum
        exp_final_easy = exp_test_easy/test_sum

        final_scores = {"rap": exp_final_rap, "rock": exp_final_rock, "easy": exp_final_easy, "country": exp_final_country}

        if winner == rock_score:
            return "Post-1980s Rock", final_scores
        elif winner == country_score:
            return "Country", final_scores
        elif winner == rap_score:
            return "Rap", final_scores
        elif winner == easy_score:
            return "Easy", final_scores

    def rap_error(self, final_scores, predictions, actual):
        max_rap_easy_error = 0
        max_rap_easy_error_i = -1

        max_rap_country_error = 0
        max_rap_country_error_i = -1

        max_rap_rock_error = 0
        max_rap_rock_error_i = -1

        for i, x in enumerate(predictions):
            if predictions[i] == "Easy" and actual[i] == "Rap":
                difference_rap = final_scores[i]["easy"] - final_scores[i]["rap"]
                if difference_rap > max_rap_easy_error:
                    max_rap_easy_error = difference_rap
                    max_rap_easy_error_i = i
            elif predictions[i] == "Post-1980s Rock" and actual[i] == "Rap":
                difference_rock = final_scores[i]["rock"] - final_scores[i]["rap"]
                if difference_rock > max_rap_rock_error:
                    max_rap_rock_error = difference_rock
                    max_rap_rock_error_i = i
            elif predictions[i] == "Country" and actual[i] == "Rap":
                difference_country = final_scores[i]["country"] - final_scores[i]["rap"]
                if difference_country > max_rap_country_error:
                    max_rap_country_error = difference_country
                    max_rap_country_error_i = i

        return {"rap_easy": max_rap_easy_error_i, "rap_rock": max_rap_rock_error_i, "rap_country": max_rap_country_error_i}

    def country_error(self, final_scores, predictions, actual):
        max_country_easy_error = 0
        max_country_easy_error_i = -1

        max_country_rap_error = 0
        max_country_rap_error_i = -1

        max_country_rock_error = 0
        max_country_rock_error_i = -1

        for i, x in enumerate(predictions):
            if predictions[i] == "Easy" and actual[i] == "Country":
                difference = final_scores[i]["easy"] - final_scores[i]["country"]
                if difference > max_country_easy_error:
                    max_country_easy_error = difference
                    max_country_easy_error_i = i
            elif predictions[i] == "Post-1980s Rock" and actual[i] == "Country":
                difference = final_scores[i]["rock"] - final_scores[i]["country"]
                if difference > max_country_rock_error:
                    max_country_rock_error = difference
                    max_country_rock_error_i = i
            elif predictions[i] == "Rap" and actual[i] == "Country":
                difference = final_scores[i]["rap"] - final_scores[i]["country"]
                if difference > max_country_rap_error:
                    max_country_rap_error = difference
                    max_country_rap_error_i = i

        return {"country_rap": max_country_rap_error_i, "country_easy": max_country_easy_error_i, "country_rock": max_country_rock_error_i}

    def rock_error(self, final_scores, predictions, actual):
        max_rock_easy_error = 0
        max_rock_easy_error_i = -1

        max_rock_rap_error = 0
        max_rock_rap_error_i = -1

        max_rock_country_error = 0
        max_rock_country_error_i = -1

        for i, x in enumerate(predictions):
            if predictions[i] == "Easy" and actual[i] == "Post-1980s Rock":
                difference = final_scores[i]["easy"] - final_scores[i]["rock"]
                if difference > max_rock_easy_error:
                    max_rock_easy_error = difference
                    max_rock_easy_error_i = i
            elif predictions[i] == "Country" and actual[i] == "Post-1980s Rock":
                difference = final_scores[i]["country"] - final_scores[i]["rock"]
                if difference > max_rock_country_error:
                    max_rock_country_error = difference
                    max_rock_country_error_i = i
            elif predictions[i] == "Rap" and actual[i] == "Post-1980s Rock":
                difference = final_scores[i]["rap"] - final_scores[i]["rock"]
                if difference > max_rock_rap_error:
                    max_rock_rap_error = difference
                    max_rock_rap_error_i = i

        return {"rock_rap": max_rock_rap_error_i, "rock_easy": max_rock_easy_error_i, "rock_country": max_rock_country_error_i}

    def easy_error(self, final_scores, predictions, actual):
        max_easy_rock_error = 0
        max_easy_rock_error_i = -1

        max_easy_rap_error = 0
        max_easy_rap_error_i = -1

        max_easy_country_error = 0
        max_easy_country_error_i = -1

        for i, x in enumerate(predictions):
            if predictions[i] == "Post-1980s Rock" and actual[i] == "Easy":
                difference = final_scores[i]["rock"] - final_scores[i]["easy"]
                if difference > max_easy_rock_error:
                    max_easy_rock_error = difference
                    max_easy_rock_error_i = i
            elif predictions[i] == "Country" and actual[i] == "Easy":
                difference = final_scores[i]["country"] - final_scores[i]["easy"]
                if difference > max_easy_country_error:
                    max_easy_country_error = difference
                    max_easy_country_error_i = i
            elif predictions[i] == "Rap" and actual[i] == "Easy":
                difference = final_scores[i]["rap"] - final_scores[i]["easy"]
                if difference > max_easy_rap_error:
                    max_easy_rap_error = difference
                    max_easy_rap_error_i = i

        return {"easy_rock": max_easy_rock_error_i, "easy_country": max_easy_country_error_i, "easy_rap": max_easy_rap_error_i}

    def calculate_statistics(self, predictions, actual):

        accuracy = 0.
        for i, x in enumerate(predictions):
            if predictions[i] == actual[i]:
                accuracy += 1

        total = len(predictions)
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

        c = Counter(actual)

        print accuracy/total
        print tabulate([['Rap', recall_rap, precision_rap, c["Rap"]], ['Country', recall_country, precision_country, c["Country"]], ['Post-1980s Rock', recall_rock, precision_rock, c["Post-1980s Rock"]], ['Easy', recall_easy, precision_easy, c["Easy"]]], headers=['Category', 'Recall', 'Precision', 'Amount'])
