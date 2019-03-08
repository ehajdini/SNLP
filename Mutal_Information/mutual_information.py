import codecs
import nltk
import math
import os
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# preprocess the text by:
# removing stop words given in the txt file
# tokenizing the text, converting the letters to lowercase
# lemmatization and stemming using nltk
def pre_process(text):
    f = codecs.open('Materials/stopwords.txt', 'r', "utf-8")
    stop_words = f.read()
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words_lemma = [lemmatizer.lemmatize(word) for word in words]
    words_stem = [stemmer.stem(word) for word in words_lemma]
    words = [w for w in words_stem if not w in stop_words]

    return words

# create the vocabulary by concatenating all training files
def create_vocabulary():
    path = 'Materials/train/'
    vocabulary = ""
    for filename in os.listdir(path):
        f = codecs.open(path + filename, 'r', "utf-8")
        text = f.read()
        vocabulary = vocabulary + text
    vocabulary = pre_process(vocabulary)
    return vocabulary


# using pointwise mutal information to select features:
# 1- use the formula to find pmi: pmi(t,category)= log(P(t,c)/ (P(t)P(c))
# 2- find the largest pmi value
# 3- select 10 features for each class
# 4- printing the features for each class
# 5- return 30 features
def calculate_PMI():
    pmi = {}

    for word in vocabulary:
        pmi1 = pmi2 = pmi3 = 0
        if word in class1:
            pmi1 = math.log((float(class1[word]) / len(tokens1)) / ((float(vocabulary[word]) / V) * (float(1) / 3)), 2)
        if word in class2:
            pmi2 = math.log((float(class2[word]) / len(tokens2)) / ((float(vocabulary[word]) / V) * (float(1) / 3)), 2)
        if word in class3:
            pmi3 = math.log((float(class3[word]) / len(tokens3)) / ((float(vocabulary[word]) / V) * (float(1) / 3)), 2)
        pmi[word] = biggest((pmi1, 1), (pmi2, 2), (pmi3, 3))

    features = {}
    features1 = {}
    features2 = {}
    features3 = {}
    c1 = c2 = c3 = 0

    key1 = lambda (key, (value, hash)): value
    for key, value in reversed(sorted(pmi.items(), key=key1)):
        if value[1] == 1 and c1 < 10:
            features1[c1] = key
            c1 += 1
            features[key] = tuple((class1[key], 1))
        if value[1] == 2 and c2 < 10:
            features2[c2] = key
            c2 += 1
            features[key] = tuple((class2[key], 2))
        if value[1] == 3 and c3 < 10:
            features3[c3] = key
            c3 += 1
            features[key] = tuple((class3[key], 3))
        if c1 == c2 == c3 == 10:
            break

    print_first10_features(features1, features2, features3)
    return features

# function built in order to print 10 feature per class
def print_first10_features(features1, features2, features3):
    print "First 10 features for 1-st class:"
    for i in features1:
        print features1[i]
    print "\n"

    print "First 10 features for 2-nd class:"
    for i in features2:
        print features2[i]
    print "\n"

    print "First 10 features for 3-rd class:"
    for i in features3:
        print features3[i]
    print "\n"

# function to find the max between three numbers, in addition it will return also the class of the max pmi value
def biggest(x, y, z):
    Max = x[0]
    cl = x[1]
    if y[0] > Max:
        Max = y[0]
        cl = y[1]
    if z[0] > Max:
        Max = z[0]
        cl = z[1]
        if y[0] > z[0]:
            Max = y[0]
            cl = y[1]
    t = (Max, cl)
    return t


# naive bayes method used to categorize:
# it will take as input the features found from previous step
# It will return the likelihood estimation
# By using:
# - feature words
# - Lidstone smoothing with alpha=0.1
def train_naive_bayes(features):
    alpha = 0.1
    likelihood = {}

    for word in features:
        if features[word][1] == 1:
            t = (word, 1)
            likelihood[tuple(t)] = math.log(float(features[word][0] + alpha) / (len(tokens1) + alpha * len(vocabulary)),
                                            2)

        if features[word][1] != 1:
            t = (word, 1)
            likelihood[tuple(t)] = math.log((alpha) / (len(tokens1) + alpha * len(vocabulary)), 2)

        if features[word][1] == 2:
            t = (word, 2)
            likelihood[tuple(t)] = math.log(float(features[word][0] + alpha) / (len(tokens2) + alpha * len(vocabulary)),
                                            2)

        if features[word][1] != 2:
            t = (word, 2)
            likelihood[tuple(t)] = math.log((alpha) / (len(tokens2) + alpha * len(vocabulary)), 2)

        if features[word][1] == 3:
            t = (word, 3)
            likelihood[tuple(t)] = math.log(float(features[word][0] + alpha) / (len(tokens3) + alpha * len(vocabulary)),
                                            2)

        if features[word][1] != 3:
            t = (word, 3)
            likelihood[tuple(t)] = math.log(float(alpha) / (len(tokens3) + alpha * len(vocabulary)), 2)

    return likelihood

# The priori probabilities,calculated as the percentage of the documents in our training set that are in each class c
# P(Biology)=P(Chemistry)=P(Physics)= 1/3
# log space is used to avoid underflow
# we will use the naive bayes classifier for 6 test files
# For each test file it will print the predicted class
def test_naive_bayes(likelihood):
    prior1 = prior2 = prior3 = math.log(float(1) / 3, 2)

    for j in [1, 2, 3, 4, 5, 6]:
        f = codecs.open('Materials/test/test' + str(j) + '.txt', 'r', "utf-8")
        text = f.read()
        tokens = pre_process(text)
        voc = frequency(tokens)
        sum1 = sum2 = sum3 = 0
        for i in voc:
            t = (i, 1)
            if i in features:
                # print i, features[i][1]
                sum1 += likelihood[tuple(t)]
                t = (i, 2)
                sum2 += likelihood[tuple(t)]
                t = (i, 3)
                sum3 += likelihood[tuple(t)]

        sum1 = prior1 + sum1
        sum2 = prior2 + sum2
        sum3 = prior3 + sum3


        file=""
        max=sum1
        file="1-st class"
        if  sum2> max:
            max = sum2
            file="2-nd class"
        if sum3 > max:
            max = sum3
            file="3-rd class"
            if sum2 > sum3:
                max = sum2
                file = "2-nd class"

        print "Test" + " " + str(j) + " corresponds to class: " + file

# This function will return frequency of each token after building the Unigrams
def frequency(tokens):
    Unigrams = {}
    for word in tokens:
        if word in Unigrams:
            Unigrams[word] += 1
        else:
            Unigrams[word] = 1
    return Unigrams


vocabulary = create_vocabulary()

f1 = codecs.open('Materials/train/Biology.txt', 'r', "utf-8")
f2 = codecs.open('Materials/train/Chemistry.txt', 'r', "utf-8")
f3 = codecs.open('Materials/train/Physics.txt', 'r', "utf-8")
text1 = f1.read()
text2 = f2.read()
text3 = f3.read()
tokens1 = pre_process(text1)
tokens2 = pre_process(text2)
tokens3 = pre_process(text3)

class1 = frequency(tokens1)
class2 = frequency(tokens2)
class3 = frequency(tokens3)
V = len(vocabulary)
vocabulary = frequency(vocabulary)
features = calculate_PMI()

likelihood = train_naive_bayes(features)

test_naive_bayes(likelihood)

