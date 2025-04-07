import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Tuple, Union
import logging

def score(text: str, 
          model: BaseEstimator, 
          vectorizer: Union[CountVectorizer, TfidfVectorizer],  
          threshold: float) -> Tuple[bool, float]:
    """
    Scores the given text based on the provided model and vectorizer, and compares 
    the predicted probability to the threshold to make a classification decision.
    
    Args:
        text (str): The input text to classify.
        model (BaseEstimator): The trained model with a `predict_proba` method.
        vectorizer (Union[CountVectorizer, TfidfVectorizer]): The vectorizer used to transform the text.
        threshold (float): The threshold to determine whether the text is classified as True (spam) or False (ham).
    
    Returns:
        Tuple[bool, float]: A tuple containing:
            - `prediction` (bool): The classification result (True for spam, False for ham).
            - `propensity` (float): The predicted probability for the positive class (spam).
    """
    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([text])
    
    # Try to predict probabilities with the model
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("The provided model must implement the 'predict_proba' method")
    
    probability = model.predict_proba(transformed_text)[0][1]  # Probability of class 1 (spam)

    # Log the probability for debugging
    logging.debug(f"Probability: {probability}")

    # Compare the probability with the threshold to get the prediction
    prediction = bool(probability >= threshold)  # Ensure the prediction is a native Python bool
    
    return prediction, float(probability)  # Ensure the probability is returned as a float