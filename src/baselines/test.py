from os import read, walk, listdir
from keyword_extractors import Yake_KE, Rake_KE
from sklearn.metrics import f1_score
from nltk.stem.porter import *

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
        stemmed_keys = [stemmer.stem(k) for k in keys]
        list_of_keys.append(stemmed_keys)

Yake_keyword_extractor = Yake_KE()
list_of_yake_answeres = [Yake_keyword_extractor.extract_stemmed_keywords(k) for k in list_of_texts] #[[(key,val)]]

print(list_of_yake_answeres[0])
print(list_of_keys[0])
print(len(list(set(list_of_yake_answeres[0]) & set(list_of_keys[0])))/ len(list_of_keys[0]))

#Rake_keyword_extractor = Rake_KE()
#print(Yake_keyword_extractor.extract_keywords(read_data))
#print(Rake_keyword_extractor.extract_keywords(read_data))
