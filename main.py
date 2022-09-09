from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import re
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

with open('model', 'rb') as f:
    model = pickle.load(f)

with open('vector', 'rb') as f:
    vect = pickle.load(f)

CLASS_LAVEL = ["Not a Hate Speech", "Hate Speech"]


@app.get("/ping")
async def ping():
    return "Hello, I think you are alive"

# Storing stopwords of english language from nltk library
sw = set(stopwords.words("english"))

# remove stop words


def filter_words(word_list):
    useful_words = [w for w in word_list if w not in sw]
    return(useful_words)


@app.post("/predict")
async def read_items(q: str | None = Query(default=None, max_length=100)):
    q = [q]
    lol = pd.DataFrame(q, columns=["tweet"])
    data = lol.copy()
    data['tl'] = [''.join([WordNetLemmatizer().lemmatize(
        re.sub('[^A-Za-z]', ' ', text)) for text in li]) for li in data['tweet']]

    a = []
    for text in data['tl']:
        word_list = word_tokenize(text)
        text = filter_words(word_list)
        a.append(text)

    train_text = []
    for i in a:
        sent = ''
        for j in i:
            sent += str(j) + ' '
        train_text.append(sent)

    data['cleaned_tweets'] = train_text

    # Transforming our data using the vector trained on training data.
    vectorized_tweets = vect.transform(data['cleaned_tweets'])

    x = model.predict(vectorized_tweets)
    return CLASS_LAVEL[x[0]]


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
