from os import read, walk, listdir
from keyword_extractors import Yake_KE, Rake_KE
from nltk.stem.porter import *
from metrics import f1_at_k
stemmer = PorterStemmer()


list_of_texts = []
filenames = listdir("../../data/SemEval2010/SemEval2010/docsutf8")
for filename in filenames:
    with open("../../data/SemEval2010/SemEval2010/docsutf8/" + filename) as f:
        list_of_texts.append(f.read())

list_of_keys = []
filenames = listdir("../../data/SemEval2010/SemEval2010/keys")
for filename in filenames:
    with open("../../data/SemEval2010/SemEval2010/keys/" + filename) as f:
        keys = f.read().splitlines()
        stemmed_keys = []
        for k in keys:
            words = k.split()
            stemmed_words = [stemmer.stem(w) for w in words]
            key = (' ').join(stemmed_words)
            stemmed_keys.append(key)
        #stemmed_keys = [stemmer.stem(k) for k in keys]
        list_of_keys.append(stemmed_keys)

Yake_keyword_extractor = Yake_KE()
Rake_keyword_extractor = Rake_KE()
list_of_yake_answeres = [Yake_keyword_extractor.extract_stemmed_keywords(k) for k in list_of_texts] #[[(key,val)]]
#list_of_rake_answeres = [Rake_keyword_extractor.extract_stemmed_keywords(k) for k in list_of_texts] #[[(key,val)]]

mean = 0
for i, pred in enumerate(list_of_yake_answeres):
    f1 = f1_at_k(pred, list_of_keys[i], k = 10)
    mean += f1 
print(mean/len(list_of_keys))

mean = 0
for i, pred in enumerate(list_of_yake_answeres):
    f1 = f1_at_k(pred, list_of_keys[i], k = 20)
    mean += f1 
print(mean/len(list_of_keys))
"""
mean = 0
for i, pred in enumerate(list_of_rake_answeres):
    f1 = f1_at_k(pred, list_of_keys[i], k = 20)
    mean += f1 
print(mean/i)
"""
#Rake_keyword_extractor = Rake_KE()
#print(Yake_keyword_extractor.extract_keywords(read_data))
#print(Rake_keyword_extractor.extract_keywords(read_data))
