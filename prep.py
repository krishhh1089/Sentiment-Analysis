
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# %%
lemmatizer = WordNetLemmatizer()

# %%
def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# %%
def lemmatize_tokens(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

# %%
import re

def advanced_negation(tokens):
    negation_words = {
        "not", "no", "never", "none", "nobody", "nothing",
        "dont", "don't", "didnt", "didn't", "doesnt", "doesn't",
        "isnt", "isn't", "wasnt", "wasn't",
        "shouldnt", "shouldn't", "cant", "can't", "cannot"
    }

    punctuation = {".", ",", "!", "?", ";", ":"}
    
    new_tokens = []
    negate = False

    for word in tokens:

        # Stop negation at punctuation
        if word in punctuation:
            negate = False
            new_tokens.append(word)
            continue

        # If current word is a negation word
        if word.lower() in negation_words:
            negate = True
            continue  # Don't add the negation word itself

        # If negation is active → prefix all words
        if negate:
            new_tokens.append("not_" + word)
        else:
            new_tokens.append(word)

    return new_tokens

# %%
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# Step 1: HTML Tag Remover
class HTMLCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [re.sub(r'<.*?>', '', text) for text in X]

# Step 2: Lowercasing + Special Char Removal
class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        result = []
        for text in X:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            result.append(text)
        return result

# Step 3: Tokenizer
class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [word_tokenize(text) for text in X]

# Step 4: Negation Handler
class NegationHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [advanced_negation(tokens) for tokens in X]

# Step 5: Stopword Remover
class StopwordRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [[w for w in tokens if w not in stop_words] for tokens in X]

# Step 6: Lemmatizer
class Lemmatizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [lemmatize_tokens(tokens) for tokens in X]

# Step 7: Token Joiner (tokens → string)
class TokenJoiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [' '.join(tokens) for tokens in X]

# Step 8: Rare Word Remover
class RareWordRemover(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=2): self.min_freq = min_freq
    def fit(self, X, y=None):
        all_tokens = ' '.join(X).split()
        counts = Counter(all_tokens)
        self.rare_words_ = {w for w, c in counts.items() if c < self.min_freq}
        return self
    def transform(self, X, y=None):
        return [' '.join(w for w in text.split() if w not in self.rare_words_) for text in X]


