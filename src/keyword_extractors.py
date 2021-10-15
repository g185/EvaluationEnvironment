
from abc import abstractmethod
import yake
from rake_nltk import Rake

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import *

from util import * 


class KeywordExtractor():
    
    def extract_keywords(self, text: str, stemming = False) -> list:
        pass

    def extract_keywords_with_weights(self, text: str, stemming = False) -> list:
        pass


class Yake_KE(KeywordExtractor):
    def __init__(self):
        self.kw_extractor = yake.KeywordExtractor()
        self.stemmer = PorterStemmer()
    
    def extract_keywords(self, text: str, stemming = False) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        unweighted_keywords = [key[0] for key in weighted_keywords]
        if stemming:
            unweighted_keywords = [stem(k) for k in unweighted_keywords]
        return unweighted_keywords
    
    def extract_keywords_with_weights(self, text: str, stemming = False) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        if stemming: 
            weighted_keywords = [(stem(k[0]), k[1]) for k in weighted_keywords]
        return weighted_keywords
    

class Rake_KE(KeywordExtractor):
    def __init__(self):
        self.kw_extractor = Rake(max_length=3)
        self.stemmer = PorterStemmer()


    def extract_keywords(self, text: str, ) -> list:
        self.kw_extractor.extract_keywords_from_text(text)
        return self.kw_extractor.get_ranked_phrases()

    def extract_stemmed_keywords(self, text: str) -> list:
        self.kw_extractor.extract_keywords_from_text(text)
        return [self.stemmer.stem(k) for k in self.kw_extractor.get_ranked_phrases()]

    def extract_keywords_with_weights(self, text: str) -> list:
        self.kw_extractor.extract_keywords_from_text(text)
        return self.kw_extractor.get_ranked_phrases_with_scores()
    
