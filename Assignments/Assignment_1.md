# Assignment 1: Prototype (due 30 Jan 2025)

## Objective:
Build a prototype for SMS spam classification.

## Tasks:

### 1. `prepare.ipynb`:
Write functions to:
- Load the data from a given file path
- Preprocess the data (if needed)
- Split the data into train, validation, and test sets
- Store the splits as `train.csv`, `validation.csv`, and `test.csv`

### 2. `train.ipynb`:
Write functions to:
- Fit a model on the training data
- Score the model on the given data
- Evaluate the model's predictions
- Validate the model:
  - Fit on the training set
  - Score on both the training and validation sets
  - Evaluate on both the training and validation sets
  - Fine-tune hyperparameters using the training and validation sets (if necessary)
- Score three benchmark models on the test data and select the best one

## Notes:
- You may download the SMS spam data from [UCI SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
- You may refer to [Radim Rehurek's Data Science Python guide](https://radimrehurek.com/data_science_python/) for building a prototype.
- For basic machine learning concepts, refer to the first three chapters of [The Elements of Statistical Learning](https://www.statlearning.com/).
- You may also refer to the **Solution Design** example covered in the class as a guideline for experiment design.
