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
print(list_of_texts[0], list_of_keys[0])

keyword_extractor1 = BartextraggoEncoder_KE("../nextraggo/experiments/pretrain/pretrain_15-11-2021_00-00-00-v1.ckpt")
keyword_extractor2 = Yake_KE(1)


list_of_yake_answeres = []
tuple = []
for t in list_of_texts:
    if len(tuple) == 2:
        list_of_yake_answeres.append(keyword_extractor1.extract_keywords(tuple, stemming = stemming))
        tuple = [] 
    tuple.append(t)
list_of_yake_answeres = [item for sublist in list_of_yake_answeres for item in sublist]

list_of_k2_answeres = [keyword_extractor2.extract_keywords(k, stemming = stemming) for k in list_of_texts] #[[(key,val)]]

print("K1", mean_f1_at_k(list_of_yake_answeres, list_of_keys[:len(list_of_yake_answeres)], k))
print("K2", mean_f1_at_k(list_of_k2_answeres, list_of_keys, k))

"""
PKE implementation

list_of_paths = []
filenames = listdir("data/" + args.dataset_name + "/" + args.dataset_name + "/docsutf8")
for filename in filenames:
    list_of_paths.append("data/" + args.dataset_name + "/" + args.dataset_name + "/docsutf8/"+filename)
list_of_yake_pke_answeres = [Yake_pke_keyword_extractor.extract_keywords(k, stemming = stemming) for k in list_of_paths]
print(mean_f1_at_k(list_of_yake_pke_answeres, list_of_keys, k))

"""
#Rake_keyword_extractor = Rake_KE()
#list_of_rake_answeres = [Rake_keyword_extractor.extract_keywords(k, stemming = stemming) for k in list_of_texts] #[[(key,val)]]
#print(mean_f1_at_k(list_of_rake_answeres, list_of_keys, k))