import yake
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')

class KeywordExtractor():
    def extract_keywords(self, text: str) -> list:
        pass


class Yake_KE(KeywordExtractor):
    def __init__(self):
        self.kw_extractor = yake.KeywordExtractor()
    
    def extract_keywords(self, text: str) -> list:
        return self.kw_extractor.extract_keywords("text")

class Rake_KE(KeywordExtractor):
        def __init__(self):
            self.kw_extractor = Rake()

        def extract_keywords(self, text: str) -> list:
            self.kw_extractor.extract_keywords_from_text(text)
            return self.kw_extractor.get_ranked_phrases_with_scores()
