from os import read, walk, listdir
from nltk.stem.porter import *

def read_texts(dataset_name: str) -> list:
    list_of_texts = []
    filenames = listdir("../data/" + dataset_name + "/" + dataset_name + "/docsutf8")
    for filename in filenames:
        with open("../data/" + dataset_name + "/" + dataset_name + "/docsutf8/" + filename) as f:
            list_of_texts.append(f.read())
    return list_of_texts

def read_paths():
    return None
def read_keywords(dataset_name: str, stemming = False) -> list:
    list_of_keys = []
    filenames = listdir("../data/" + dataset_name + "/" + dataset_name + "/keys")
    for filename in filenames:
        with open("../data/" + dataset_name + "/" + dataset_name + "/keys/" + filename) as f:
            keys = f.read().splitlines()
            if stemming:
                keys = [stem(k) for k in keys]             
            list_of_keys.append(keys)
    return list_of_keys

def read_texts_and_keywords(dataset_name: str, stem_keywords = False):
    return read_texts(dataset_name), read_keywords(dataset_name, stem_keywords)

def stem(text: str, all_words = True) -> str:
    stemmer = PorterStemmer()
    if all_words == True:
        words = text.split()
        stemmed_words = [stemmer.stem(w) for w in words]
        text = (' ').join(stemmed_words)
    else:
        text = stemmer.stem(text)
    return text

