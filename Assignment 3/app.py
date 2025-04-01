from flask import Flask, request, jsonify
import joblib
from score import score
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # This allows you to log debug messages
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load pre-trained model and vectorizer
try:
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    logging.info("Model and vectorizer successfully loaded")
except FileNotFoundError as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")
    raise e

@app.route('/score', methods=['POST'])
def score_endpoint():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if 'text' is provided in the request
        if not data or 'text' not in data:
            logging.warning("No text provided in the request")
            return jsonify({'error': 'No text provided in the request'}), 400
        
        text = data['text']

        # Get the prediction and propensity using the score function
        prediction, propensity = score(text, model, vectorizer, threshold=0.5)

        # Return the response in JSON format
        logging.info(f"Prediction: {prediction}, Propensity: {propensity}")
        return jsonify({
            'prediction': prediction,
            'propensity': propensity
        })
    
    except Exception as e:
        # Catch any exception and return it as an error response
        logging.error(f"Error processing the request: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)  