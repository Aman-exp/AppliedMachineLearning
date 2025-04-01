# Import libraries
from sklearn.base import BaseEstimator as estimator
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required nltk datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> list[str]:
    """
    Function to preprocess text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): The text to preprocess.
        
    Returns:
        list[str]: The preprocessed tokens.
    """
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Converting all tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Removing empty strings
    tokens = [token for token in tokens if token != '']
    
    return tokens

def score(text: str, model: estimator, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Function to score a trained model on a given text input.
    
    Args:
        text (str): The input text to score.
        model (estimator): The trained model to use for scoring.
        threshold (float): The threshold for making a binary prediction (default is 0.5).
        
    Returns:
        tuple[bool, float]: The prediction (True/False) and the propensity score.
        
    Raises:
        ValueError: If the input text is not a string, the model is not an estimator, 
                    or the threshold is outside the range [0, 1].
    """
    # Validate inputs
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    if not isinstance(model, estimator):
        raise ValueError("Model must be an instance of sklearn BaseEstimator")
    
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")
    
    # Preprocess the text (tokenize, remove stopwords, and lemmatize)
    tokens = preprocess_text(text)
    # Converting list of tokens back to string for model prediction (you might need a vectorizer here)
    text_for_model = ' '.join(tokens)
    
    # Get the propensity score (probability for the positive class)
    propensity = model.predict_proba([text_for_model])[0][1]

    # Make a prediction based on the threshold
    prediction = propensity >= threshold

    return prediction, propensity