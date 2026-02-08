"""
    Author: Ashuwin P
    Email: ashuwin2210335@ssn.edu.in
    File Description:
    This module implements a robust BM25L (Okapi BM25 with length normalization) text retrieval
    model wrapper using the rank_bm25 library. It supports text preprocessing, building the BM25L
    index on a corpus, scoring queries against the corpus, and saving/loading the model state.

    Usage:
    - Instantiate the RobustBM25L class with a corpus (list of documents as strings).
    - Use get_scores(query) to score and rank corpus documents for a given query string.
    - Use save(filepath) to serialize the model and load(filepath) to load a saved model.

    The implementation follows PEP 8 style for comments and line-length, using clear and
    concise block comments to describe functions and critical code sections.
"""

import math
import re
import pickle
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25L

# Ensure the English stopwords corpus from NLTK is available, download if missing
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


class RobustBM25L:
    """
    A robust BM25L information retrieval implementation that preprocesses corpus documents,
    builds an efficient BM25L index, and provides fast scoring of queries.

    Attributes:
        corpus (List[str]): Original list of document texts.
        k1 (float): Term frequency saturation parameter (default 1.5).
        b (float): Length normalization parameter (default 0.75).
        stemmer_language (str): Language used for stemming (English).
        stemmer (SnowballStemmer): Stemmer instance for word normalization.
        stop_words (set): Set of stopwords to exclude during preprocessing.
        processed_corpus (List[List[str]]): Tokenized and stemmed corpus.
        bm25 (BM25L): Instance of the BM25L model from rank_bm25 library.
        corpus_size (int): Number of documents in corpus.
        doc_len (List[int]): Lengths of processed documents.
        avgdl (float): Average document length in the corpus.
        idf (dict): Document frequency map for terms (computed on processed corpus).
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize the RobustBM25L instance with the given corpus and BM25L hyperparameters.
        Preprocess the corpus and build the BM25L index.

        Args:
            corpus (List[str]): List of raw text documents.
            k1 (float): BM25L k1 parameter, controls term frequency saturation.
            b (float): BM25L b parameter, controls document length normalization.
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Set up English stopwords and stemmer for text preprocessing
        self.stemmer_language = "english"
        self.stemmer = SnowballStemmer(self.stemmer_language)
        self.stop_words = set(stopwords.words(self.stemmer_language))

        # Internal storage for tokenized/stemmed documents
        self.processed_corpus = []

        # Build the BM25L model index on the processed corpus
        self._initialize()

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------
    def _preprocess(self, text: str) -> List[str]:
        """
        Preprocess input text into a list of stemmed tokens after lowercasing,
        removing stopwords, filtering short tokens, and tokenizing with regex.

        Args:
            text (str): Raw text string to preprocess.

        Returns:
            List[str]: List of processed tokens/stems.
        """
        tokens = re.findall(r"\w+(?:'\w+)?", text.lower())

        return [
            self.stemmer.stem(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]

    # -----------------------------------------
    # Build index (tokenization + BM25L init)
    # -----------------------------------------
    def _initialize(self):
        """
        Preprocess the entire corpus and initialize the BM25L ranking model instance.
        Also computes corpus statistics like document lengths and document frequencies
        for additional metadata and compatibility.
        """
        self.processed_corpus = [self._preprocess(doc) for doc in self.corpus]

        # Initialize BM25L model from the external library using processed corpus
        self.bm25 = BM25L(self.processed_corpus, k1=self.k1, b=self.b)

        # Store corpus statistics
        self.corpus_size = len(self.corpus)
        self.doc_len = [len(doc) for doc in self.processed_corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))

        # Compute document frequency (DF) map for terms (for informational use)
        df_map = {}
        for tokens in self.processed_corpus:
            for t in set(tokens):
                df_map[t] = df_map.get(t, 0) + 1
        self.idf = df_map

    # -----------------------------------------
    # Scoring using BM25L library
    # -----------------------------------------
    def get_scores(self, query: str):
        """
        Compute BM25L relevance scores for each document in the corpus given a query string.

        Args:
            query (str): Query text string.

        Returns:
            List[float]: List of BM25L scores aligned with corpus documents.
        """
        processed = self._preprocess(query)
        if not processed:
            # Return zero scores if query preprocessing yields no valid tokens
            return [0.0] * self.corpus_size

        return self.bm25.get_scores(processed)

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    def save(self, filepath: str):
        """
        Serialize the model metadata and processed corpus to a binary file.

        Args:
            filepath (str): Path to save the pickled model data.
        """
        data = {
            "corpus": self.corpus,
            "k1": self.k1,
            "b": self.b,
            "processed_corpus": self.processed_corpus,
            "stop_words": list(self.stop_words),
            "stemmer_language": self.stemmer_language,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    # -----------------------------------------
    # Load model
    # -----------------------------------------
    @classmethod
    def load(cls, filepath: str):
        """
        Load a previously saved model from a file and reconstruct its state,
        including reinitialization of the BM25L model and preprocessing metadata.

        Args:
            filepath (str): Path of the saved pickled model file.

        Returns:
            RobustBM25L: Restored model instance.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Create uninitialized instance
        obj = cls.__new__(cls)

        # Restore metadata fields
        obj.corpus = data["corpus"]
        obj.k1 = data["k1"]
        obj.b = data["b"]

        obj.stemmer_language = data["stemmer_language"]
        obj.stemmer = SnowballStemmer(obj.stemmer_language)
        obj.stop_words = set(data["stop_words"])

        obj.processed_corpus = data["processed_corpus"]

        # Rebuild BM25L model on processed corpus
        obj.bm25 = BM25L(obj.processed_corpus, k1=obj.k1, b=obj.b)

        obj.corpus_size = len(obj.corpus)
        obj.doc_len = [len(doc) for doc in obj.processed_corpus]
        obj.avgdl = sum(obj.doc_len) / max(1, len(obj.doc_len))

        # Recompute document frequency map for terms
        df_map = {}
        for tokens in obj.processed_corpus:
            for t in set(tokens):
                df_map[t] = df_map.get(t, 0) + 1
        obj.idf = df_map

        return obj
