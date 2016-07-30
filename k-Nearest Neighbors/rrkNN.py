##########################################
# ross-ranney k-Nearest Neighbors class.
#######################################

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter

from tabulate import tabulate


class rrkNN(object):

    def __init__(self, X_train_data, Y_train_data, X_test_data, Y_test_data, k):
        self.X_train_data = X_train_data
        self.Y_train_data = Y_train_data
        self.k = k
        self.X_test_data = X_test_data
        self.Y_test_data = Y_test_data

        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems

    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item.encode('utf-8').decode('utf-8', "replace")))
        return stemmed

    def train_corpus(self):
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        self.X_train = vectorizer.fit_transform(self.X_train_data)

    def predict_song(self, song):
        cosine_similarities = linear_kernel(self.X_train, song).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-self.k:-1]
        words_to_count = self.Y_train_data[related_docs_indices]
        c = Counter(words_to_count)
        return c.most_common(1)[0][0]

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
        print tabulate([['Rap', recall_rap, precision_rap, c["Rap"]], ['Country', recall_country, precision_country, c["Country"]], ['Post 1980s Rock', recall_rock, precision_rock, c["Post-1980s Rock"]], ['Easy', recall_easy, precision_easy, c["Easy"]]], headers=['Category', 'Recall', 'Precision', 'Amount'])
