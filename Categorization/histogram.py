import glob
import nltk
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

#the function plots the histogram for given author text, takes as input the location of files
def histogram(path):
    # first, opens and reads all the files for an author and concatenates the text 
    files = glob.glob(path)
    temp = []
    for file in files:    
        with open(file, "r") as data:        
            temp.append(data.read())

    text= ''.join(temp)
     
    #tokenize and reduce to lowercase   
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]  
    #optional step to remove stop words
    stop_words = set(stopwords.words('english'))  
    words = [w for w in words if not w in stop_words]
       
    #create word-frequency dictionary and take top 10 most common words
    counts = dict(Counter(words).most_common(10))    
    
    #set the plot parameters
    labels, values = zip(*counts.items())
    bar_width = 0.35
    indexes = np.arange(len(labels))
    plt.bar(indexes, values)
    # add labels
    plt.xticks(indexes + bar_width, labels)
    plt.show()



#call function with the locations of the files
histogram('james/*.txt')
histogram('london/*.txt')
