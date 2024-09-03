import os
from flask import Flask, request, jsonify
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch.nn.functional as F
from data_preprocess import clean_text  # Import the preprocessing functions

app = Flask(__name__)

# Define a dictionary mapping labels to class names
label_to_class = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load model and tokenizer at startup
MODEL_DIR = "models/fine_tuned_igbo_sentiment"

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"Model and tokenizer loaded successfully. Using device: {DEVICE}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

def prepare_input(text, max_length=512):
    """
    Preprocess the text and tokenize it for the model.
    """
    # Preprocess the text using the clean_text function
    cleaned_text = clean_text(text)
    
    # Tokenize the cleaned text
    encoding = tokenizer(
        cleaned_text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return encoding

def predict(text):
    """
    Make a prediction using the trained model and return both the predicted class and probability.
    """
    inputs = prepare_input(text)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to logits to get probabilities
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the predicted class and its probability
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_class_name = label_to_class.get(predicted_class_id, "unknown")
    predicted_class_prob = probabilities[0, predicted_class_id].item()
    
    return {
        "label": predicted_class_name,
        "probability": round(predicted_class_prob, 4)
    }

@app.route('/')
def home():
    """
    Home route that returns a basic welcome message.
    """
    return jsonify({"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to get predictions."})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint to get predictions.
    Expects JSON input with a 'text' field.
    """
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415  # Unsupported Media Type

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({
            "error": "Invalid input. Please provide a JSON object with a 'text' field."
        }), 400
    
    text = data['text']
    
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "The 'text' field must be a non-empty string."
        }), 400
    
    try:
        result = predict(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5009))
    app.run(host='0.0.0.0', port=port)