import pickle
import os


def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer from pickle files."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer