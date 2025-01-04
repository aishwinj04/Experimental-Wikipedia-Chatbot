import nltk
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas

# initialize 
lemmatizer = WordNetLemmatizer()
text = 'Originally, vegetables were collected from the wild by hunter-gatheres. Vegetables are all plants. Vegetables can be eaten either cooked or raw.'
question = 'What are vegetables?'


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


# tokenize sentence
sentence_tokens = nltk.sent_tokenize(text)

# calculate word frequency
tv = TfidfVectorizer(tokenizer=lemma_me)
tf = tv.fit_transform(sentence_tokens)
print(tf)

# rows for each sentence, columns = unique words
df = pandas.DataFrame(tf, columns=tv.get_feature_names_out())
print(df)