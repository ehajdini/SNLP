
import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt


#calculate the dictionaries from text
def word_dict(text):
    # Post: return a list of words ordered from the most frequent to the least frequent
    tokens = nltk.word_tokenize(text)
    freq  = dict()       
    for word in tokens:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1    
    return sorted(freq,reverse=True) #values sorted in reverse



def find_OOV(tokens,dictionary,size): 
    top_dict = dictionary[:size]   #choose the size the dictionary as per value passed to function, take top 'size' values
    unseen = 0
    for word in tokens:
        if word not in top_dict:
            unseen +=1    
    OOV_rate = unseen/len(tokens)     
    return OOV_rate



#files = {"train1.txt","train2.txt","train3.txt","train4.txt","train5.txt"}
files = {"train5.txt"}
  
for f in files:
    file = open(f,"r")   
    dictionary=word_dict(file.read())    #call function to calculate dictionary
    #print(dictionary)    
    
    #open and read the tokens in test file 
    testfile = open("test.txt", "r")
    test_text = testfile.read()
    tokens = nltk.word_tokenize(test_text)    
       
    OOV = find_OOV(tokens, dictionary,len(dictionary)) #call function with the value of size as the full dictionary
    print ("For file ", f, " OOV is " , OOV) 
    
    # to store the OOV values with varying sizes 
    OOV_rate = []        
    for i in range(1,len(dictionary)-1):
        OOV_rate.append(find_OOV(tokens, dictionary, i)) #call function with varying values of dictionary size
        
        
    plt.plot(OOV_rate)  
    plt.xlabel("size of vocabulary")
    plt.ylabel("OOV rate")
    plt.ylim((0.4,1))
    plt.title(f)
    plt.pause(0.05)
    
    plt.show()  

    
        
		
oov = [0.5163331066899223, 0.5356963372453957,0.42368522186419133,0.286221066012357,0.2057823632009933]
size = [325,449,704,2210,3700]   

plt.scatter(size,oov)
plt.xlabel("size of vocabulary")
plt.ylabel("OOV rate") 
plt.show() 
