import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
import spacy
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
pd.set_option('display.max_colwidth', None)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables parallel processing

def make_ngram(texts):
    bigram_phrases = Phrases(texts, min_count=5, threshold=10)
    trigram_phrases = Phrases(bigram_phrases[texts], threshold=10)
    bigram = Phraser(bigram_phrases)
    trigram = Phraser(trigram_phrases)
    ngrams_docs = []
    for doc in trigram[bigram[texts]]:
        filtered_ngrams = [token for token in doc if "_" in token]
        ngrams_docs.append(filtered_ngrams)
    return ngrams_docs

def text_to_bow(text):
    return id2word.doc2bow(text)

def dominant_topic (l) :
    return max(l, key=lambda x: x[1])[0]