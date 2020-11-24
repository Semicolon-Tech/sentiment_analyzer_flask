import os
import string
import random

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB

import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

wordnet_lemmatizer: WordNetLemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    # removes upper cases
    text = text.lower()
    
    # removes punctuation
    for char in string.punctuation:
        text = text.replace(char, "")
    
    # lemmatize the words and join back into string text
    text = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    
    return text


class DenseTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x.todense()
    
    def __str__(self):
        return "DenseTransformer()"
    
    def __repr__(self):
        return self.__str__()
    
    
class CleanTextTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self
    
    @staticmethod
    def transform(x, y=None, **fit_params):
        return np.vectorize(clean_text)(x)

    def __str__(self):
        return 'CleanTextTransformer()'

    def __repr__(self):
        return self.__str__()


def load_imdb_sentiment_analysis_dataset(imdb_data_path, seed=123):
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        print(f"loading train: {category} ...")
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in tqdm(sorted(os.listdir(train_path))):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname), encoding="utf-8") as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        print(f"loading test: {category} ...")
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in tqdm(sorted(os.listdir(test_path))):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname), encoding="utf-8") as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((np.array(train_texts), np.array(train_labels)),
            (np.array(test_texts), np.array(test_labels)))


class CategoricalBatchNB(TransformerMixin):
    def __init__(self, batch_size, classes, *args, **kwargs):
        self._batch_size = batch_size
        self._classes = classes
        self._args = args
        self._kwargs = kwargs
        self._model = CategoricalNB(*args, **kwargs)

    def fit(self, x, y, **fit_params):
        batch_size = self._batch_size
        self._model = CategoricalNB(*self._args, **self._kwargs)

        for index in tqdm(range(batch_size, x.shape[0] + batch_size, batch_size)):
            self._model.partial_fit(
                x[index - batch_size:index, :].toarray(),
                y[index - batch_size:index],
                classes=self._classes
            )
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x

    def predict(self, x):
        batch_size = self._batch_size
        predictions = []
        for index in tqdm(range(batch_size, x.shape[0] + batch_size, batch_size)):
            predictions.extend(
                self._model.predict(
                    x[index - batch_size:index, :].toarray()
                ).tolist()
            )
        return np.array(predictions).ravel()

    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

    def __str__(self):
        return "CategoricalBatchNB()"

    def __repr__(self):
        return self.__str__()


class GaussianBatchNB(TransformerMixin):
    def __init__(self, batch_size, classes, *args, **kwargs):
        self._batch_size = batch_size
        self._classes = classes
        self._args = args
        self._kwargs = kwargs
        self._model = GaussianNB(*args, **kwargs)
        
    def fit(self, x, y, **fit_params):
        batch_size = self._batch_size
        self._model = GaussianNB(*self._args, **self._kwargs)
        
        for index in tqdm(range(batch_size, x.shape[0]+batch_size, batch_size)):
            self._model.partial_fit(
                x[index-batch_size:index, :].toarray(),
                y[index-batch_size:index], 
                classes=self._classes
            )                  
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x
    
    def predict(self, x):
        batch_size = self._batch_size
        predictions = []
        for index in tqdm(range(batch_size, x.shape[0]+batch_size, batch_size)):
            predictions.extend(
                self._model.predict(
                    x[index-batch_size:index, :].toarray()
                ).tolist()
            )
        return np.array(predictions).ravel()
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

    def __str__(self):
        return "GaussianBatchNB()"

    def __repr__(self):
        return self .__str__()