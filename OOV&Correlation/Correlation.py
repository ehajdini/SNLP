import nltk
import math
import matplotlib.pyplot as plt

list = {"y'all", "y'alls", "y'all'd", "y'all'd've", 
         "y'all're",  "y'all've",  "you'd", "you'd've",
          "you'll", "you'll've", "you're", "you've", "your" 
        }

alice = nltk.corpus.gutenberg.raw('carroll-alice.txt')
#print(alice)

for word in list:
    alice = alice.lower().replace(word, "you") 

#write to file output.txt       
fout = open("output.txt", "w")    
print (alice, file=fout)


word ="you"
tokens = alice.split()  #all tokens in text
N=len(tokens)   #the total number of token ie total size of text corpus

#find all the positions in text where the word "you" occurs
positions = [i for i, w in enumerate(tokens) if w == word]
#print(positions)

#the total number of times the word "you" occurs in text
wordcount= len(positions)

count = {}
correlation = []

#If the distance between 2 consecutive positions of "you" matches d+1, i.e. d tokens between them, then increment count[d]
for d in range(1,50):
    count[d] = 0
    for i in range(0,len(positions)-1):
        if (positions[i+1]-positions[i]==d+1):
            count[d]=count[d]+1
    #print (count[d])
    correlation.append((count[d]/wordcount)/math.pow((wordcount/N),2))
    
                
plt.plot(correlation)  
plt.xlabel("distance")
plt.ylabel("correlation")
plt.show()         
    
    

    

