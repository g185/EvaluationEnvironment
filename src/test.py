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

Yake_keyword_extractor = BartextraggoEncoder_KE("../nextraggo/experiments/pretrain/pretrain_15-11-2021_00-00-00-v1.ckpt")
list_of_yake_answeres = [Yake_keyword_extractor.extract_keywords(k, stemming = stemming) for k in list_of_texts] #[[(key,val)]]

print(mean_f1_at_k(list_of_yake_answeres, list_of_keys, k))

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