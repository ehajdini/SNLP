import codecs
from operator import itemgetter
import nltk
from nltk.corpus import stopwords
import math

# There are two functions to preprocess the text:
# 1-st Ignoring stop words
# 2-nd Including stop words
# Removing punctuation and tokenizing the text
def pre_process(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    tokens = [word.lower() for word in words if word.isalpha()]
    copy_tokens = [w for w in tokens if not w in stop_words]
    return copy_tokens

def pre_process_without_stop_words(text):
    words = nltk.word_tokenize(text)
    tokens = [word.lower() for word in words if word.isalpha()]
    return tokens

# Concatenating documents into their respective classes: 1-st class (james), 2-nd class(london)
# Building the vocabulary V, consisting of the union of all the word types in all classes
# The function will return 2 classes and vocabulary

def create_classes():
    text1 = text2 = ""
    for i in [1, 2, 3]:
        f = codecs.open('james/Henry James___' + str(i) + '.txt', 'r', "utf-8")
        text1 = text1 + f.read()
    james_tokens = pre_process(text1)

    for i in [1, 2, 3]:
        f = codecs.open('london/Jack London___' + str(i) + '.txt', 'r', "utf-8")
        text2 = text2 + f.read()
    london_tokens = pre_process(text2)

    vocabulary = text1 + text2
    vocabulary = pre_process(vocabulary)
    return james_tokens, london_tokens, vocabulary


# This function will take as parameter tokens of 1-st & 2-nd class, and vocabulary
# It will return the likelihood estimation for each word of the vocabulary
# By using:
# - the frequency of words in each class
# - Lidstone smoothing with alpha=0.1
def train_naive_bayes(class1, class2, vocabulary):
    alpha = 0.1
    likelihood = {}

    freq1 = frequency(class1)
    freq2 = frequency(class2)
    vocabulary = frequency(vocabulary)

    for word in vocabulary:
        if word in class1:
            t = (word, 1)
            likelihood[tuple(t)] = math.log(float(freq1[word] + alpha) / (len(class1) + alpha * len(vocabulary)), 2)
        if word not in class1:
            t = (word, 1)
            likelihood[tuple(t)] = math.log(float(alpha) / (len(class1) + alpha * len(vocabulary)), 2)
        if word in class2:
            t = (word, 2)
            likelihood[tuple(t)] = math.log(float(freq2[word] + alpha) / (len(class2) + alpha * len(vocabulary)), 2)
        if word not in class2:
            t = (word, 2)
            likelihood[tuple(t)] = math.log(float(alpha) / (len(class2) + alpha * len(vocabulary)), 2)

    return likelihood


# This function will return frequency of each token after building the Unigrams
def frequency(tokens):
    Unigrams = {}
    for word in tokens:
        if word in Unigrams:
            Unigrams[word] += 1
        else:
            Unigrams[word] = 1
    return Unigrams


# This function will take as input the likelihood of each word and Vocabulary
# for each word of the testing file, if it is in vocabulary we will find its respective likelihood estimation
# The priori probabilities,calculated as the percentage of the documents in our training set that are in each class c
# P(james)=P(london)= (3/6)= 0.5
# log space is used to avoid underflow
# we will use the naive bayes classifier for 3 test files
# For each test file it will print the predicted class
def test_naive_bayes(likelihood, voc):

    prior1 = prior2 = math.log(0.5, 2)

    for j in [1]:
        f = codecs.open('test/test' + str(j) + '.txt', 'r', "utf-8")
        text = f.read()
        tokens = pre_process(text)
        sum1 = sum2 = 0
        for i in tokens:
            t = (i, 1)
            if i in voc:
                sum1 += likelihood[tuple(t)]

        for i in tokens:
            t = (i, 2)
            if i in voc:
                sum2 += likelihood[tuple(t)]

        sum1 = prior1 + sum1
        sum2 = prior2 + sum2

        if sum1 > sum2:
            c = "James"
        if sum1 < sum2:
            c = "London"
            print ("Test" + " " + str(j) + " corresponds to class: " + c)


def printfirst10(C):
    D = frequency(C)
    top10_keys = {}
    top10_values = {}
    c = 0

    for key, value in reversed(sorted(D.items(), key=itemgetter(1))):
        top10_keys[c] = key
        top10_values[c] = value
        print (key, value)
        c += 1
        if c == 10:
            break
    print("\n")


c1, c2, voc = create_classes()
likelihood = train_naive_bayes(c1, c2, voc)
test_naive_bayes(likelihood, voc)
