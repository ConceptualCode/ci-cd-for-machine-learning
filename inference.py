import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer


label_to_class = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def load_model_and_tokenizer(model_dir):
    """
    Load the fine-tuned model and tokenizer from the specified directory.
    """
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def prepare_input(text, tokenizer, max_length=512):
    """
    Tokenize and prepare the input text for the model.
    """
    encoding = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    return encoding

def predict(text, model, tokenizer, device):
    """
    Make a prediction using the trained model.
    """
    model.eval()
    inputs = prepare_input(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

     # Apply softmax to logits to get probabilities
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the predicted label
    predicted_class_id = logits.argmax().item()
    predicted_class_prob = probabilities[0, predicted_class_id].item()
    predicted_class_name = label_to_class[predicted_class_id]

    return predicted_class_id, predicted_class_name, predicted_class_prob

def main():
    model_dir = "results/checkpoint-1911"  # Directory where the model is saved

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Please make sure you have a GPU-enabled environment.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.to(device)
    
    # Example input text for inference
    text = "Ihe a na-ewute m"
    
    # Get prediction
    prediction, predicted_class, probability = predict(text, model, tokenizer, device)
    
    # Print the predicted label
    print(f"Predicted label for the input text: {prediction}")
    print(f"Predicted class of the input text: {predicted_class}")
    print(f"Probability of the predicted label: {probability:.4f}")

if __name__ == "__main__":
    main()