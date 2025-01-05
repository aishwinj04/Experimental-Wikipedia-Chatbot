# Experimental Wikipedia Chatbot via Command Line

### Overview

This Python program uses Natural Language Processing (NLP) techniques to fetch relevant information from Wikipedia based on a user's question about a given topic. It tokenizes the input question and Wikipedia content, then calculates the similarity between the question and each sentence in the article using **TF-IDF** and **cosine similarity**. The program will output the most relevant sentence or indicate if no suitable response is found.

This system leverages several libraries:
- **NLTK** for tokenization and lemmatization.
- **Scikit-learn** for text vectorization and calculating similarity.
- **Wikipedia API** to fetch Wikipedia page content.
- **Pandas** for handling and displaying text data.

### Features

1. **Topic Selection**: The user is prompted to input a topic (such as "Python programming"), and the program fetches the Wikipedia page content for that topic.
2. **Question Answering**: The user can ask questions about the topic. The program processes the question and the article to find the most relevant sentence.
3. **Lemmatization**: The system lemmatizes words (e.g., turning "running" into "run") to improve the accuracy of similarity calculations.
4. **Cosine Similarity**: It calculates the cosine similarity between the question and each sentence in the Wikipedia page. The sentence with the highest similarity is returned as the answer.
5. **Exit Command**: The user can type `q` to exit the program at any time.

### Dependencies

Before running the program, ensure that the following Python libraries are installed:

- `nltk`
- `scikit-learn`
- `pandas`
- `wikipedia`

### Limitations and Notes
Experimental Nature: This system is experimental and may not always provide accurate or complete answers.
Dependency on Wikipedia: Answers depend on the content available on the Wikipedia page for the given topic. If the page is sparse or the question is too specific, the program might fail to provide a relevant answer.
Accuracy: The similarity threshold is currently set at 0.2. If the highest cosine similarity is below this threshold, the program will not return an answer.
Performance: The performance of this program may degrade with longer Wikipedia pages or more complex queries.



 
