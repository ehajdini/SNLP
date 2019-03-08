import codecs
import nltk
import math
from operator import itemgetter

f1 = codecs.open('data/train.txt', 'r', "utf-8")
f2 = codecs.open('data/test.txt', 'r', "utf-8")
text = f1.read()
test = f2.read()


# Removing punctuation and tokenizing the text
def pre_process(text):
    words = nltk.word_tokenize(text)
    tokens = [word.lower() for word in words if word.isalpha()]
    return tokens


# this function calculates and returns the probabilities of unigrams and frequency of each word
def unigram_prob():
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
# It also returns the count of bigrams
def bigram_prob(Unigrams):
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

    Bigram_prob = {}
    for word in Bigram:
        Bigram_prob[tuple(word)] = float(Bigram[word]) / Unigrams[word[0]]
        # print str(float(Bigram[word]) / Unigrams[word[0]])+" "+str(word)+" "+str(Bigram[word])+" "+str(Unigrams[word[0]])

    return Bigram_prob, Bigram


# this function takes as input the count of bigrams.
# It calculates & returns the conditional probabilities of trigrams
# It also returns their count
def trigram_prob(Bigrams):
    Trigram = {}
    Trigram_prob = {}
    trigram_tuples = tuple(nltk.trigrams(tokens))
    for triple in trigram_tuples:
        if triple in Trigram:
            Trigram[triple] += 1
        else:
            Trigram[triple] = 1

    # pruning trigrams that occur more than once
    Trigram_copy = dict(Trigram)
    for (key, value) in Trigram_copy.items():
        if value == 1:
            del Trigram[key]

    # computing conditional probabilities of trigrams
    for triple in Trigram:
        Trigram_prob[tuple(triple)] = float(Trigram[triple]) / Bigrams[(triple[0], triple[1])]

    return Trigram_prob, Trigram


# takes as input parameters the Trigram's counts and Bigram's counts from training text
def smooth3(Trigrams, Bigrams):
    smoothedP = {}


    #smooth trigrams using laplace smoothing and our data from training data set
    for triple in test_trigrams:
        if triple in Trigrams:
            smoothedP[tuple(triple)] = (float(Trigrams[triple] + 0.1)) / (Bigrams[(triple[0], triple[1])] + V * 0.1)
        elif triple not in Trigrams and (triple[0], triple[1]) in Bigrams:
            smoothedP[tuple(triple)] = float(0.1) / (Bigrams[(triple[0], triple[1])] + V * 0.1)
        else:
            smoothedP[tuple(triple)] = 0.1 / (V * 0.1)

    return smoothedP

# Following the model for trigram probabilities, we will need to use the conditional probabilities for
# unigrams,bigrams and trigrams we found before
def interpolated():
    Lambda = float(1) / 3
    P_3 = {}
    for triple in test_trigrams:

        if triple in Trigrams_prob:
            P_3[tuple(triple)] = Lambda * Unigram_prob[triple[0]] + \
                                 Lambda * Bigram_prob[(triple[0], triple[1])] + Lambda * Trigrams_prob[triple]

        elif triple not in Trigrams_prob and (triple[0], triple[1]) in Bigram_prob:
            P_3[tuple(triple)] = Lambda * Unigram_prob[triple[0]] + \
                                 Lambda * Bigram_prob[(triple[0], triple[1])]

        elif (triple[0], triple[1]) not in Bigram_prob and triple[0] in Unigram_prob:
            P_3[tuple(triple)] = Lambda * Unigram_prob[triple[0]]
        else:
            P_3[tuple(triple)] = 0

    return P_3


# Firstly, we find all trigrams in testing file
# Then we prune the trigrams that occur only once
# After that we compute the relative frequency of trigrams
def trigram_compute_rel_freq():
    Trigram = {}
    Trigram_freq = {}
    trigram_tuples = tuple(nltk.trigrams(test_tokens))
    for triple in trigram_tuples:
        if triple in Trigram:
            Trigram[triple] += 1
        else:
            Trigram[triple] = 1

    # pruning trigrams that occur only once
    Trigram_copy = dict(Trigram)
    for (key, value) in Trigram_copy.items():
        if value == 1:
            del Trigram[key]

    N = len(Trigram)
    for triple in Trigram:
        Trigram_freq[tuple(triple)] = float(Trigram[triple]) / N

    return Trigram_freq, Trigram


# this function takes as input relative frequencies from testing set and conditional probabilities from training set
# It will return the Perplexity value
def perpelexity(relative_freq, conditional_prob):
    PP = 0
    for triple in test_trigrams:
        if conditional_prob[triple] != 0:
            PP += relative_freq[triple] * math.log(conditional_prob[triple], 2)
    return -PP

# Function used to print first 10 trigrams
def printfirst10(D):
    c = 0
    for key, value in reversed(sorted(D.items(), key=itemgetter(1))):
        print key, value
        c += 1
        if c == 10:
            break
    print("\n")


#preprocessing
tokens = pre_process(text)
test_tokens = pre_process(test)

# computing the unsmoothed unigram, bigram and trigram probabilities
Unigram_prob, unigram_counts = unigram_prob()
Bigram_prob, bigram_counts = bigram_prob(unigram_counts)
Trigrams_prob, trigram_counts = trigram_prob(bigram_counts)

# vocabulary length
V = len(unigram_counts)

# find relative frequencies in testing text
relative_freq, test_trigrams = trigram_compute_rel_freq()

# smoothed trigram probabilities with Laplace Smoothing
smoothed_P3 = smooth3(trigram_counts, bigram_counts)

# smoothed trigram probabilities using interpolation
P_3 = interpolated()

print "Perplexity using 1-st Model:"
print perpelexity(relative_freq, smoothed_P3)
print "Perplexity using 2-nd Model:"
print perpelexity(relative_freq, P_3)

# calling this function will print 10 trigrams with highest probabilities with respect to training data
printfirst10(smoothed_P3)
printfirst10(P_3)