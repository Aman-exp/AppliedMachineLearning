# Coverage Report

To generate a coverage report and save it to a file named `coverage.txt`, follow the steps below:

### Steps:

1. Open your terminal or command line interface.
2. Navigate to the root directory of your project.
3. Run the following command:

   ```bash
   pytest --cov=score --cov=test --cov-report=term-missing > coverage.txt
