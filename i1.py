import nltk

document = "The morning sun warmed my skin as I walked to the local coffee shop, where the aroma of freshly brewed beans greeted me. The weather was clear, and the park bench felt comfortable under my hands."

nltk.download('stopwords')
nltk.download('punkt_tab')

def remove_stop_words(doc: str) -> str:
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.tokenize.word_tokenize(doc)
    return " ".join([token for token in tokens if token not in stopwords])

def stemming(doc: str) -> str:
    stemmer = nltk.stem.SnowballStemmer('english')
    tokens = nltk.tokenize.word_tokenize(doc)
    return " ".join([stemmer.stem(token) for token in tokens])

rem_stop = remove_stop_words(document)
stemmed = stemming(rem_stop)
print(f"{rem_stop=}")
print(f"{stemmed=}")
