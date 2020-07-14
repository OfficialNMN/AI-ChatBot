import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bagofwords(tokenized_sentence,words):
    sentence_words=[stem(word) for word in tokenized_sentence]
    bag=np.zeros(len(words),dtype=np.float32)
    for id,w in enumerate(words):
        if w in sentence_words:
            bag[id]=1
    return bag



