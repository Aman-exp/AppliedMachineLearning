import pytest
import joblib
from score import score
import requests
import subprocess
import time
import os
import signal

# Fixtures to load model and vectorizer
@pytest.fixture
def model():
    return joblib.load('best_model.pkl')

@pytest.fixture
def vectorizer():
    return joblib.load('vectorizer.pkl')

# Unit tests for the scoring function
def test_score(model, vectorizer):
    # Test basic prediction and propensity type
    prediction, propensity = score("Test message", model, vectorizer, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

    # Repeat to check output types again
    prediction, propensity = score("Another test", model, vectorizer, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

    # Test propensity value bounds (0 <= propensity <= 1)
    _, propensity = score("Test message", model, vectorizer, 0.5)
    assert 0 <= propensity <= 1

    # Test with threshold 0 (always predict spam)
    prediction, _ = score("Any message", model, vectorizer, 0)
    assert prediction == True

    # Test with threshold 1 (always predict non-spam)
    prediction, _ = score("Any message", model, vectorizer, 1)
    assert prediction == False

    # Test for obvious spam message
    obvious_spam = """Congratulations! You've won a $1000 gift card! To claim your prize, just click the link and provide your details: www.fakeprize.com"""
    prediction, propensity = score(obvious_spam, model, vectorizer, 0.5)
    assert prediction == True
    assert propensity > 0.5

    # Test for obvious non-spam message
    obvious_ham = "Hey Sarah, let's catch up this weekend! How about Saturday afternoon?"
    prediction, propensity = score(obvious_ham, model, vectorizer, 0.5)
    assert prediction == False
    assert propensity < 0.5


# Helper function to wait for Flask app to start
def wait_for_flask(timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.get('http://localhost:5000/')
            return True
        except requests.ConnectionError:
            time.sleep(0.1)
    return False

# Integration tests for Flask app
def test_flask():
    try:
        # Start the Flask app using os.system()
        process = subprocess.Popen(
            ["flask", "run", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for Flask to be ready
        if not wait_for_flask():
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Flask app failed to start or become ready.\nStdout: {stdout}\nStderr: {stderr}")
        
        # Send POST request with test message
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': 'Test message'}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'propensity' in data
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['propensity'], float)
        assert 0 <= data['propensity'] <= 1

        # Test with obvious spam message
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': """Congratulations! You've won a $1000 gift card! To claim your prize, just click the link and provide your details: www.fakeprize.com"""}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] == True
        assert data['propensity'] > 0.5

        # Test with obvious ham message
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': "Hey Sarah, let's catch up this weekend! How about Saturday afternoon?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] == False
        assert data['propensity'] < 0.5

        # Test invalid request (missing text)
        response = requests.post(
            'http://localhost:5000/score',
            json={}
        )
        assert response.status_code == 400

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise

    finally:
        # Ensure the Flask process is terminated after the test
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()