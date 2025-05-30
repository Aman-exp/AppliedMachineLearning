# Assignment 4: Containerization & Continuous Integration (due 8 Apr 2025)

## 1. containerization
- create a docker container for the flask app created in Assignment 3
- create a Dockerfile which contains the instructions to build the container, which include
    - installing the dependencies
    - copying app.py and score.py
    - launching the app by running “python app.py” upon entry
- build the docker image using Dockerfile
- run the docker container with appropriate port bindings
- in test.py write test_docker(..) function which does the following
    - launches the docker container using commandline (e.g. os.sys(..), docker build and docker run)
    - sends a request to the localhost endpoint /score (e.g. using requests library)
    - for a sample text
    - checks if the response is as expected
    - close the docker container
In coverage.txt, produce the coverage report using pytest for the tests in test.py
## 2. continuous integration
- write a pre-commit git hook that will run the test.py automatically every time you try to commit the code to your local ‘main’ branch
- copy and push this pre-commit git hook file to your git repo

References
- [https://docker-curriculum.com/](https://docker-curriculum.com/)
- [https://www.tutorialspoint.com/docker/docker_overview.htm](https://www.tutorialspoint.com/docker/docker_overview.htm)


- [https://www.freecodecamp.org/news/how-to-dockerize-a-flask-app/](https://www.freecodecamp.org/news/how-to-dockerize-a-flask-app/)


- [https://githooks.com/](https://githooks.com/)
- [https://www.atlassian.com/git/tutorials/git-hooks](https://www.atlassian.com/git/tutorials/git-hooks)

- [https://www.giacomodebidda.com/posts/a-simple-git-hook-for-your-python-projects/](https://www.giacomodebidda.com/posts/a-simple-git-hook-for-your-python-projects/)