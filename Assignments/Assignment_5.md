# Assignment 4: Assignment 5: Transfer Learning (due date 17 April 2025) [extended to 19 April 2025]

## 1. Transfer Learning for image data using CNN

- Download about 100 images of chickens and 100 images of ducks from the internet
- In a google colab notebook, fine-tune a pre-trained convolutional neural network to classify duck vs chicken and output the classification report

- [https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/](https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/)

- [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

- [https://www.learnpytorch.io/](https://www.learnpytorch.io/)

- build the docker image using Dockerfile
- run the docker container with appropriate port bindings
- in test.py write test_docker(..) function which does the following
    - launches the docker container using commandline (e.g. os.sys(..), docker build and docker run)
    - sends a request to the localhost endpoint /score (e.g. using requests library)
    - for a sample text
    - checks if the response is as expected
    - close the docker container
In coverage.txt, produce the coverage report using pytest for the tests in test.py
## 2. Transfer Learning for text data using Transformer

- Download the sentiment analysis dataset from
[https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- Build a sentiment analysis classifier to classify the sentiment into positive, neutral, and negative by fine-tuning a pre-trained transformer model and print your classification report

[https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/temp/sentiment-analysis-using-bert-keras-movie-reviews.html](https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/temp/sentiment-analysis-using-bert-keras-movie-reviews.html)

[https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/)
