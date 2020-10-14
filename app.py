import joblib
import sklearn

import numpy as np
from flask import Flask, request

# utilities
from utils import clean_text

app = Flask(__name__)

models = {
    "multinomial": {
        "count": joblib.load("models/multinomial_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/multinomial_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "bernoulli": {
        "count": joblib.load("models/bernoulli_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/bernoulli_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "complement": {
        "count": joblib.load("models/complement_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/complement_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
}


@app.route('/predict', methods=["POST"])
def predict():
    # all the necessary parameters to select the right model
    parameters = request.json

    # the parameters to select model
    model = parameters.pop("model")
    vectorizer = parameters.pop("vectorizer")
    text = parameters.pop("text")

    x = [text]  # the input
    y = models[model][vectorizer].predict(x)  # prediction

    # the final response to send back
    response = "positive" if y else "negative"
    return response


@app.route('/predict_all', methods=["POST"])
def predict_all():
    # all the neccesary parameters to select the right model
    parameters = request.json

    # the parameters to selet model
    text = parameters.pop("text")

    # the final response to send back
    response = {}

    x = [text]  # the input
    for model in models:
        response[model] = {}

        for vectorizer in models[model]:
            y = models[model][vectorizer].predict(x)  # prediction
            response[model][vectorizer] = "positive" if y else "negative"

    return response


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()
