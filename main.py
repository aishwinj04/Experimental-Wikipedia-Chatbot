import nltk
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# initialize 
lemmatizer = WordNetLemmatizer()

# lemmatize words in sentence
def lemma_me(sentence):
    sentence_tokens = nltk.word_tokenize(sentence.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, tag in zip(sentence_tokens, pos_tags):
        if tag[1][0].lower() in ['n','v','a','r']:
            lemma = lemmatizer.lemmatize(token, tag[1][0].lower())
            sentence_lemmas.append(lemma)

    return sentence_lemmas


lst = lemma_me('The quick brown fox jumped over the lazy dog')
print(lst)
