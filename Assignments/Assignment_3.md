# Assignment 3: Testing & Model Serving (Due: 1st April 2025)

## Overview

In this assignment, you will implement unit tests for a scoring function in `score.py`, test it using pytest, and deploy a Flask application to serve predictions from a trained model. The goal is to write tests that check the functionality and integration of your model serving pipeline.

## Tasks

### 1. Unit Testing (`score.py`)

- In `score.py`, write a function with the following signature to score a trained model on a text input:

  ```python
  def score(text: str, model: sklearn.estimator, threshold: float) -> (prediction: bool, propensity: float):
- In test.py, write a unit test function test_score(...) to test the score function.
You may reload and use the best model saved during experiments in train.ipynb (in joblib/pkl format) for testing the score function.
You may consider the following points to construct your test cases:
  - does the function produce some output without crashing (smoke test)
  - are the input/output formats/types as expected (format test)
  - is prediction value 0 or 1 
  - is propensity score between 0 and 1
  - if you put the threshold to 0 does the prediction always become 1
  - if you put the threshold to 1 does the prediction always become 0
  - on an obvious spam input text is the prediction 1 
  - on an obvious non-spam input text is the prediction 0

### 2. Flask serving
- In app.py, create a flask endpoint /score that receives a text as a POST request and gives a response in the json format consisting of prediction and propensity
- In test.py, write an integration test function test_flask(...) that does the following:
  -  launches the flask app using command line (e.g. use os.system)
  -  test the response from the localhost endpoint
  -  closes the flask app using command line

In coverage.txt produce the coverage report output of the unit test and integration test using pytest

[https://docs.pytest.org/en/8.0.x/](https://docs.pytest.org/en/8.0.x/)
[https://flask.palletsprojects.com/en/2.3.x/quickstart/](https://flask.palletsprojects.com/en/2.3.x/quickstart/)
