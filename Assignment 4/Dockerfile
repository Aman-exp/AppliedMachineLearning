FROM python:3.9-slim

WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv /env
ENV PATH="/env/bin:$PATH"

# Copy application files
COPY app.py /app/
COPY score.py /app/
COPY best_model.pkl /app/
COPY vectorizer.pkl /app/
COPY requirements.txt /app/

# Install dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]