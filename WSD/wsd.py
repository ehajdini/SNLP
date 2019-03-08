import glob
import os
import nltk
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score

#this function takes the text as input,
#performs pre-processing including stop word removal, lowercase, remove punctuation, stemming
#returns tokens list

def preprocess(text):    
    text.translate(string.punctuation)
    tokens = nltk.word_tokenize(text)       
    stop_words = set(stopwords.words('english'))  
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [token.lower() for token in tokens if token.isalpha()] 
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens ]      
    #exclude = set(string.punctuation)
    #tokens = ''.join(ch for ch in tokens if ch not in exclude)    
    return tokens


#read the files, store definitions and test text
def read_test_sentences(word):
    word_text = []
    word_labels = []    
    filename = "wsd_data/"+word+".test"
    with open(filename, "r+") as lines:                       
        for line in lines:
            line= line.rstrip('\n')
            if line.startswith("#LABEL"):
                label = line.split(" ")[1]
                word_labels.append(label)
                text = next(lines).rstrip('\n')
                word_text.append(text)
                word_labeled_text= pd.DataFrame({'label' :word_labels, 'text': word_text}) 
        
        return (word_labeled_text)
    
    
def read_definitions(word):
    word_sense = {}
    filename = "wsd_data/"+word+".definition"   
    #print(filename)    
    with open(filename, "r+") as lines:
        for line in lines:
            line= line.rstrip('\n')
            #print(line) 
            if line.startswith("#DEFINITION"):
                sense = line.split(" ")[1]
                meaning = next(lines).rstrip('\n')
                word_sense[sense] = meaning
                
        #print(word_sense) 
    lines.close()
    
    #find most frequent sense    
    with open(filename) as f:
        first_line = f.readline().rstrip('\n')
        most_frequent_sense = first_line.split(" ")[1]  
        #print(most_frequent_sense)        
    return word_sense, most_frequent_sense 
  
                
def similarity_score(X,Y):
    #X_Y = [token for token in X if token in Y]
    XandY = list(set(X) & set(Y))      
    similarity = 2* len(XandY) / ( len(X) + len(Y))    
    return similarity
         
  
def Jaccard_similarity_score(X,Y):      
    XandY = list(set(X) & set(Y))        
    XorY = list(set(X).union(Y))       
    similarity =  len(XandY) / len(XorY)    
    return similarity
    
#calculates the accuracy of the predicted values when compared with the actual word senses            
def get_accuracy(test_data, predicted_sense_list):
    actual_label_list = test_data['label'].tolist()    
#     print(actual_label_list)
#     print (predicted_sense_list)        
    accuracy=accuracy_score(actual_label_list,predicted_sense_list)
    return str(round(accuracy, 3))


#calculates the accuracy of the predicted values when compared with the baseline ie when the word sense is the most frequent one   
def get_baseline_accuracy(most_frequent_sense, predicted_sense_list): 
    most_frequent_sense_list = []
    for i in range(0, len(predicted_sense_list)):
        most_frequent_sense_list.append(most_frequent_sense) 
    accuracy=accuracy_score(most_frequent_sense_list,predicted_sense_list)
    #accuracy = sum(1 for x,y in zip(most_frequent_sense_list,predicted_sense_list) if x == y) / len(predicted_sense_list)        
    return str(round(accuracy, 3))

    
# calculates the word sense based on the similarity scores using sim1 formula     
def similarity(test_data, definitions): 
    senses = list()        
    similarity = {} 
    text_list = test_data['text'].tolist()
    for text in text_list:
        #print(text)
        X = preprocess(text)            
        #calculate the scores for the senses of word           
        for sense, meaning in definitions.items():
            Y = preprocess(meaning)            
            similarity[sense] = similarity_score(X,Y)  
        # choose the sense with higher score                   
        word_sense = max(similarity, key=similarity.get)
        #print ("Word sense is " +word_sense+ "\n")
        senses.append(word_sense)    
    return senses        
                          
# calculates the word sense based on the similarity scores using jaccard formula 
def Jaccard_similarity(test_data, definitions):
        Jaccard_senses = list()        
        Jaccard_similarity = {}        
        text_list = test_data['text'].tolist()
        for text in text_list:
            #print(text)
            X = preprocess(text)  
            for sense, meaning in definitions.items():                
                Y = preprocess(meaning) 
                Jaccard_similarity[sense] = Jaccard_similarity_score(X,Y) 
            word_sense_jaccard = max(Jaccard_similarity, key=Jaccard_similarity.get)
            #print ("Word sense with Jaccard_similarity is " +word_sense_jaccard+ "\n")
            Jaccard_senses.append(word_sense_jaccard)            
        return Jaccard_senses
           
    
#this function defines the data path, then calls the read_test_sentences() and read_definitions() to load the data
#after this, the similarities are computed for the test data using both the similarity formulae
#lastly, the accuracy values are reported
          
def Lesk():
    testpath = "wsd_data/*.test"    
    files = glob.glob(testpath)    
    for file in files:         
        word = os.path.basename(file).split(".")[0]  
        #print(word)                      
        #load test data and definitions and most frequent sense for each word
        test_data = read_test_sentences(word)  
        definitions, most_frequent = read_definitions(word)     
        
        senses = similarity(test_data, definitions)   
        jaccard_senses = Jaccard_similarity(test_data, definitions)    
         
        print ("For word "  + word )
        accuracy = get_accuracy(test_data, senses) 
        print ("Accuracy is :" )
        print(accuracy) 
                
        baseline_accuracy = get_baseline_accuracy(most_frequent, senses)  
        print ("Baseline Accuracy is :" )
        print(baseline_accuracy) 
                  
        print ("For word "  + word )
        Jaccuracy = get_accuracy(test_data, jaccard_senses)               
        print ("Accuracy is :" )
        print(Jaccuracy) 
                
#         Jbaseline_accuracy = get_baseline_accuracy(most_frequent, jaccard_senses)  
#         print ("Baseline Accuracy is :" )
#         print(Jbaseline_accuracy)

        
Lesk()        
        
        
        

   
        