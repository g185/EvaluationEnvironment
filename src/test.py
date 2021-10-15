
from keyword_extractors import *
from metrics import *
from util import *



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
