import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from evaluate import load
import json


def compute_metrics(eval_pred):
    # Load the evaluation metrics
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    precision_metric = load("precision")
    recall_metric = load("recall")

    # Extract logits and labels
    logits, labels = eval_pred
    # Get predictions by taking the argmax of the logits
    predictions = np.argmax(logits, axis=-1)

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Return a dictionary of metrics
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

def evaluate_model():
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the fine-tuned model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained("models/fine_tuned_igbo_sentiment")
    tokenizer = RobertaTokenizer.from_pretrained("models/fine_tuned_igbo_sentiment")
    
    # Move model to GPU
    model.to(device)
    
    # Load the test dataset
    test_dataset = load_dataset('csv', data_files='preprocessed_data/test.csv')
    
    # Tokenize the test dataset
    test_dataset = test_dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", max_length=512, truncation=True), batched=True)
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate the model
    results = trainer.evaluate(eval_dataset=test_dataset['train']) 
    
    # Print the evaluation results
    print(f"Test set evaluation results: {results}")
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    evaluate_model()