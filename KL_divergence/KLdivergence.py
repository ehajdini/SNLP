import codecs
import nltk
import math

f_1 = codecs.open("Ludwig_Anzengruber_-_Der_Sternsteinhof_(1885).txt","r","utf-8")
f_2 = codecs.open("austen-persuasion.txt", "r", "utf-8" )
f_3 = codecs.open("austen-emma.txt","r","utf-8")


text_1=f_1.read()
text_2=f_2.read()
text_3=f_3.read()


#to perform tokenization and lowercase for text
def pre_process(text):
    frequency = {}
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]    
#    word_dict ={}    
    for word in words:
        count = frequency.get(word, 0)
        frequency[word] = count + 1    
#        word_dict[word] = words.count(word)    
    return frequency,len(words)


def Entropy(frequency,N):
    pi_vec = [float(i) / N for i in reversed(sorted(frequency.values()))]
    H=0
    for p in pi_vec:
        H += p * math.log(p, 2)
    return -H

def Lidstone_smoothing(word_dict, V, N):    
    alpha =0.1
    count = [i for i in reversed(sorted(word_dict.values()))]      
    p_lidstone = {}    
    
    for w in V:
        if w in count:
            p_lidstone[w] = float((count[w]+alpha)/(N + alpha * len(V)))
        else:
            p_lidstone[w] = float(alpha/(N + alpha * len(V)))  
    return p_lidstone


def KL_divergence(text_1,text_2):    
    f1,N1=pre_process(text_1)
    f2,N2=pre_process(text_2)
    
    #compute union of vocabulary 
    vocab = [w for w in f1]
    for token in f2:
        if token not in vocab:
            vocab.append(token)
            
 #   print(len(vocab))  
          
    # perform smoothing of both texts with a vocabulary equal to union of both texts      
    smooth_text1 = Lidstone_smoothing(f1,vocab,N1)
    smooth_text2 = Lidstone_smoothing(f2,vocab,N2)    
    
    divergence=0
    for i in vocab:
        pi = smooth_text1[i]
        qi = smooth_text2[i]   
        divergence = divergence + pi*math.log(pi/qi, 2) 
    return divergence


f1,tokens1=pre_process(text_1)
f2,tokens2=pre_process(text_2)
f3,tokens3=pre_process(text_3)

h1=Entropy(f1,tokens1)
h2=Entropy(f2,tokens2)
h3=Entropy(f3,tokens3)

print ("Entropy of german text 1: ",h1)
print ("Entropy of english text 2: ",h2)
print ("Entropy of english text 3: ",h3)
 
print ("KL divergence German Text1 and English Text2 : ", KL_divergence(text_1,text_2)) 
print ("KL divergence German Text1 and English Text3 : ", KL_divergence(text_1,text_3)) 
print ("KL divergence English Text2 and English Text3 : ",KL_divergence(text_2,text_3))

 

