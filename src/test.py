import argparse
from keyword_extractors import *
from metrics import *
from util import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("-stem", action="store_true", help="Interactive mode")
    parser.add_argument("-k", type=int, default=10, help="Num sequences")
    # return
    return parser.parse_args()

args = parse_args()
stemming = args.stem
k = args.k
list_of_texts, list_of_keys = read_texts_and_keywords(dataset_name=args.dataset_name, stem_keywords=stemming)

Yake_keyword_extractor = Yake_KE()

list_of_yake_answeres = [Yake_keyword_extractor.extract_keywords(k, stemming = stemming) for k in list_of_texts] #[[(key,val)]]

print(mean_f1_at_k(list_of_yake_answeres, list_of_keys, k))

#Rake_keyword_extractor = Rake_KE()
#list_of_rake_answeres = [Rake_keyword_extractor.extract_keywords(k, stemming = stemming) for k in list_of_texts] #[[(key,val)]]
#print(mean_f1_at_k(list_of_rake_answeres, list_of_keys, k))