import codecs
import nltk
import matplotlib.pyplot as plt
import math
import numpy as np

f1 = codecs.open('data/kfold.txt', 'r', "utf-8")
text = f1.read()

# Removing punctuation and tokenizing the text
def pre_process(text):
    words = nltk.word_tokenize(text)
    tokens = [word.lower() for word in words if word.isalpha()]
    return tokens


# this function calculates and returns the probabilities of unigrams and frequency of each word
def unigram_prob(tokens):
    prob_vector = {}
    N = len(tokens)
    Unigrams = {}
    for word in tokens:
        if word in Unigrams:
            Unigrams[word] += 1
        else:
            Unigrams[word] = 1

    for word in Unigrams:
        prob_vector[word] = float(Unigrams[word]) / N

    return prob_vector, Unigrams

# this function takes as input the number of unigrams.
# It calculates & returns the conditional probabilities of bigrams
def bigram_prob(Unigrams, train_tokens):
    Bigram = createBigrams(train_tokens)

    Bigram_prob = {}
    for word in Bigram:
        Bigram_prob[tuple(word)] = float(Bigram[word]) / Unigrams[word[0]]
    return Bigram_prob


# this function generates the bigrams taking tokens as input
def createBigrams(tokens):
    bigram_tuples = tuple(nltk.bigrams(tokens))
    Bigram = {}
    for couple in bigram_tuples:
        if couple in Bigram:
            Bigram[couple] += 1
        else:
            Bigram[couple] = 1

        # create a temporary dictionary and prune bigrams which occur only once
    Bigram_copy = dict(Bigram)
    for (key, value) in Bigram_copy.items():
        if value == 1:
            del Bigram[key]

    return Bigram

#Computing the relative frequencies of bigrams
def compute_rel_freq(Bigrams):
    Bigram_freq={}
    N = len(Bigrams)
    for words in Bigrams:
        Bigram_freq[words] = float(Bigrams[words]) / N

    return Bigram_freq

# this function takes as input relative frequencies from testing set and conditional probabilities from training set
# It will return the Perplexity value
def perpelexity(test,relative_freq, conditional_prob):
    PP = 0
    for words in test:
        if conditional_prob[words] != 0:
            PP += relative_freq[words] * \
                  math.log(conditional_prob[words], 2)
    return -PP


# this function will find the optimal perplexity value while changing the lambda value in bigram model
def find_optimal_values(test,train,num):

    uni_prob, uni_counts = unigram_prob(train)
    bi_prob = bigram_prob(uni_counts, train)
    test_bigrams = createBigrams(test)
    rel_freq = compute_rel_freq(test_bigrams)

    PP={}
    for i in np.arange(0.01, 1.0, 0.01):
        i= round(i,2) #two digit precision
        P2 = {}
        for words in test_bigrams:
            if words in bi_prob:
                P2[words] = i * uni_prob[words[0]] + (1 - i) * bi_prob[words]
            elif words not in bi_prob and words[0] in uni_prob:
                P2[words] = i * uni_prob[words[0]]
            else:
                P2[words] = 0
        PP[i]=perpelexity(test_bigrams, rel_freq, P2)

        Lambda = min(PP.keys(), key=(lambda k: PP[k]))
        min_PP=PP[Lambda]

    # plotting the perplexity-lambda graph
    lists = sorted(PP.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.xlabel('Lambda')
    plt.ylabel('Perplexity')
    plt.savefig('plot'+str(num)+'.png')
    plt.show()

    return min_PP,Lambda


def cross__fold(d):
    avg = len(tokens) / float(d)
    last = 0.0

    # creating 5 folds of data
    folds = {}
    c = 0
    while last < len(tokens):
        folds[c] = (tokens[int(last):int(last + avg)])
        c += 1
        last += avg

    sum=0

    print("Perplexity","Lambda 1","Lambda 2")

    test = folds[0]
    train = folds[1] + folds[2] + folds[3] + folds[4]
    minPP,Lambda=find_optimal_values(test,train,0)
    print(minPP,Lambda,(1-Lambda))
    sum+=minPP

    test = folds[1]
    train = folds[0] + folds[2] + folds[3] + folds[4]
    minPP, Lambda = find_optimal_values(test, train,1)
    print(minPP, Lambda,(1-Lambda))
    sum+=minPP

    test = folds[2]
    train = folds[0] + folds[1] + folds[3] + folds[4]
    minPP, Lambda = find_optimal_values(test, train,2)
    print(minPP, Lambda,(1-Lambda))
    sum+=minPP

    test = folds[3]
    train = folds[0] + folds[1] + folds[2] + folds[4]
    minPP, Lambda = find_optimal_values(test, train,3)
    print(minPP, Lambda,(1-Lambda))
    sum+=minPP

    test = folds[4]
    train = folds[0] + folds[1] + folds[2] + folds[3]
    minPP, Lambda = find_optimal_values(test, train,4)
    print(minPP, Lambda,(1-Lambda))
    sum+=minPP

    avgPP=float(sum)/5
    return avgPP


tokens = pre_process(text)
averagePP=cross__fold(5)
print "\n Average Perplexity: " + str(averagePP)