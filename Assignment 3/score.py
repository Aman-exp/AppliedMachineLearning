import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

def score(text: str, 
          model: BaseEstimator, 
          vectorizer: CountVectorizer,  # or TfidfVectorizer
          threshold: float) -> Tuple[bool, float]:
    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([text])
    
    # Try to predict probabilities with the model
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(transformed_text)[0][1]
        print(probability)
    else:
        raise ValueError("The provided model must implement the 'predict_proba' method")
    
    # Compare the probability with the threshold to get the prediction
    prediction = bool(probability >= threshold)  # Ensure the prediction is a native Python bool
    
    return prediction, float(probability)  # Ensure the probability is returned as a float