
import yake
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import *


class KeywordExtractor():
    def extract_keywords(self, text: str) -> list:
        pass

    def extract_keywords_with_weights(self, text: str) -> list:
        pass


class Yake_KE(KeywordExtractor):
    def __init__(self):
        self.kw_extractor = yake.KeywordExtractor()
        self.stemmer = PorterStemmer()
    
    def extract_keywords(self, text: str) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        unweighted_keywords = [key[0] for key in weighted_keywords]
        return unweighted_keywords
    
    def extract_stemmed_keywords(self, text: str) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        unweighted_keywords = [self.stemmer.stem(key[0]) for key in weighted_keywords]
        return unweighted_keywords

    def extract_keywords_with_weights(self, text: str) -> list:
        return self.kw_extractor.extract_keywords(text)
    

class Rake_KE(KeywordExtractor):
    def __init__(self):
        self.kw_extractor = Rake()
        self.stemmer = PorterStemmer()

    def extract_keywords(self, text: str) -> list:
        self.kw_extractor.extract_keywords_from_text(text)
        return self.kw_extractor.get_ranked_phrases_with_scores()
    
    def extract_keywords_with_weights(self, text: str) -> list:
        pass