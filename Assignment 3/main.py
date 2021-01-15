import random
import codecs
import string
import gensim
import matplotlib.pyplot as plt

from nltk import FreqDist
from nltk.stem.porter import PorterStemmer


# Returns paragraphs read from file, separated by the given separator
def readParagraphs(file, separator):
    paragraphs = []
    currentParagraph = ""
    for line in file:
        if line == separator:
            # If end of paragraph, add paragraph to collection
            if currentParagraph:
                paragraphs.append(currentParagraph)
                currentParagraph = ""
        else:
            # Add line to current paragraph
            currentParagraph += line
    return paragraphs


# Removes text punctuation and white characters, converts to lower-case and stems words in paragraphs
def preprocess(paragraph):
    # Translation table is used to remove punctuation and white characters
    translationTable = dict.fromkeys(map(ord, string.punctuation + "\n\r\t"), None)
    stemmer = PorterStemmer()

    return [stemmer.stem(word.translate(translationTable).lower()) for word in paragraph]


# Returns term frequency for a given paragraph
def getTermFrequency(paragraph, word):
    freqDist = FreqDist(paragraph)
    return freqDist[word]


if __name__ == '__main__':

    # TASK 1

    # Random numbers generator
    random.seed(123)

    # Load file using UTF-8 encoding
    file = codecs.open("pg3300.txt", "r", "utf-8")

    # Read paragraphs from file
    paragraphs = readParagraphs(file, "\r\n")

    # Filter out paragraphs containing the word "Gutenberg"
    paragraphs = [paragraph for paragraph in paragraphs if "gutenberg" not in paragraph.lower()]

    # Split paragraphs into list of words
    tokenizedParagraphs = [paragraph.split() for paragraph in paragraphs]

    # Preprocess all words (remove punctuation/white characters, lower-case, stem)
    processedParagraphs = [preprocess(paragraph) for paragraph in tokenizedParagraphs]

    # Test word frequency function for paragraph 9 with the word "is"
    print("Word frequency for word \"is\" in paragraph 9:", getTermFrequency(processedParagraphs[8], "is"), "\n")

    # Printing results from task 1
    print("TASK 1 RESULTS (FIRST 5 PARAGRAPHS):\n[")
    for i in range(5):
        print(processedParagraphs[i], ",")
    print("...]\n")

    # TASK 2

    # Create dictionary
    dictionary = gensim.corpora.Dictionary(processedParagraphs)

    # Read file containing stopwords
    file = codecs.open("common-english-words.txt", "r", "utf-8")

    # Putting all stopwords inside array (and stem them)
    stemmer = PorterStemmer()
    stopwords = [stemmer.stem(word) for word in file.read().split(",")]

    # Get ids for all stopwords found in dictionary
    stopwordIds = []
    for stopword in stopwords:
        stopwordId = dictionary.token2id.get(stopword)
        if stopwordId is not None:
            stopwordIds.append(stopwordId)

    # Filter stopwords from dictionary
    dictionary.filter_tokens(stopwordIds)

    # Mapping paragraphs into Bags-of-Words
    corpus = [dictionary.doc2bow(paragraph) for paragraph in processedParagraphs]

    # Printing results from task 2
    print("TASK 2 RESULTS (FIRST 5 PARAGRAPHS):\n[")
    for i in range(5):
        print(corpus[i], ",")
    print("...]\n")

    # TASK 3

    # Building TF-IDF model
    tfidfModel = gensim.models.TfidfModel(corpus)

    # Mapping Bags-of-Words into TF-IDF weights
    tfidfCorpus = tfidfModel[corpus]

    # Constructing MatrixSimilarity object
    tfidfIndex = gensim.similarities.MatrixSimilarity(tfidfCorpus)

    # Building LSI model
    lsiModel = gensim.models.LsiModel(tfidfCorpus, id2word=dictionary, num_topics=100)

    # Creating corpus using LSI model
    lsiCorpus = lsiModel[tfidfCorpus]

    # Constructing MatrixSimilarity object
    lsiIndex = gensim.similarities.MatrixSimilarity(lsiCorpus)

    # Printing first 3 LSI topics
    topics = lsiModel.show_topics(3)
    print("TASK 3 - First 3 LSI topics:")
    for topic in topics:
        print(topic[1])

    # TASK 4

    # Query specified in task description
    query = "What is the function of money?"

    # Remove punctuations, tokenize and stem
    query = preprocess(query.split())

    # Convert query to Bags-of-Words representation
    query = dictionary.doc2bow(query)

    # Convert Bags-of-Words to TF-IDF representation
    tfidfQuery = tfidfModel[query]

    # Print TF-IDF weights for query
    print("\nTASK 4 - TF-IDF weights for query:")
    for t in tfidfQuery:
        print(dictionary[t[0]], ":", t[1])

    # Find top 3 most relevant paragraphs according to TF-IDF model and print them
    doc2similarity = enumerate(tfidfIndex[tfidfQuery])
    tfidfTop3 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print("\nTASK 4 - Top 3 paragraphs according to TF-IDF model:\n")
    for t in tfidfTop3:
        print("[ paragraph", t[0] + 1, "]")
        paragraph = paragraphs[t[0]].split("\r\n")
        for (i, line) in enumerate(paragraph):
            if i < 5:
                print(line)
        print()

    # Convert query TF-IDF representation to LSI-topics representation
    lsiQuery = lsiModel[query]

    # Find top 3 topics with most significant weights and print them
    topics = sorted(lsiQuery, key=lambda kv: -abs(kv[1]))[:3]
    print("TASK 4 - Top 3 topics according to LSI model:")
    for topic in topics:
        print("\n[ topic", topic[0], "]")
        print(lsiModel.show_topics()[topic[0]])

    # Find top 3 relevant paragraphs according to LSI model and print them
    doc2similarity = enumerate(lsiIndex[lsiQuery])
    lsiTop3 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print("\nTASK 4 - Top 3 paragraphs according to LSI model:\n")
    for t in lsiTop3:
        print("[ paragraph", t[0] + 1, "]")
        paragraph = paragraphs[t[0]].split("\r\n")
        for (i, line) in enumerate(paragraph):
            if i < 5:
                print(line)
        print()

    # PLOT GRAPH FOR FREQUENCY DISTRIBUTION OF TOP 15 WORDS

    # Count frequency of each word using corpus
    wordFrequencies = [(0, 0)] * len(dictionary)
    for p in corpus:
        for i in p:
            wordFrequencies[i[0]] = (i[0], wordFrequencies[i[0]][1] + i[1])

    # Convert word id into strings using dictionary
    wordFrequency = [(dictionary[wf[0]], wf[1]) for wf in wordFrequencies]

    # Sort to find top 15
    top15words = sorted(wordFrequency, key=lambda wf: -wf[1])[:15]

    # Draw top 15 as a bar chart
    plt.bar([i[0] for i in top15words], [i[1] for i in top15words])
    plt.xlabel("Top 15 words")
    plt.ylabel("Number of occurences")
    plt.legend()
    plt.show()