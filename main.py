import nltk
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas
import warnings

warnings.filterwarnings("ignore", message=".*token_pattern.*")


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


# tokenize text and find similarity 
def find_similar(text, question):
    # tokenize text
    sentence_tokens = nltk.sent_tokenize(text) # split into sentences
    sentence_tokens.append(question) # add the question 

    # calculate word importance 
    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)

    # rows for each sentence, columns = unique words
    df = pandas.DataFrame(tf.toarray(), columns=tv.get_feature_names_out())

    # find similarity between each sentence in text and the question
    values = cosine_similarity(tf[-1], tf)

    # argsort for indices that would sort the array 
    # most similar at second last position (max is the question itself)
    index = values.argsort()[0][-2]  


    values_flat = values.flatten()
    values_flat.sort() 

    coef = values_flat[-2] 
    if coef > 0.3:
       # print(sentence_tokens[index])  -2 represents highest similarity which is at index 1 of original list 
        return sentence_tokens[index]
    

def main():
    text = 'Originally, vegetables were collected from the wild by hunter-gatheres. Vegetables are all plants. Vegetables can be eaten either cooked or raw.'
    while True:
        question = input("Hi, What do u want to know? ")
        output = find_similar(text, question)
        print(output)
        


if __name__ == '__main__':
    main()
