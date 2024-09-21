import json
import optuna
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import numpy as np
import torch
from evaluation import compute_metrics


def objective(trial):

    # if not torch.cuda.is_available():
    #     print("CUDA is not available, Use CPU instead")
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Suggest hyperparameters for this trial
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 1)
    per_device_batch_size = trial.suggest_categorical('per_device_batch_size', [8, 16, 32])

    # Load dataset and model
    dataset = DatasetDict({
        'train': load_dataset('csv', data_files='preprocessed_data/train.csv')['train'],
        'validation': load_dataset('csv', data_files='preprocessed_data/val.csv')['train']
    })

    num_labels = len(set(dataset['train']['label']))

    model_name = "checkpoint-1520000"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.to(device)

    # Tokenize datasets
    max_length = 128
    dataset = dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", truncation=True, max_length=max_length), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./pm_search",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./pm_search_logs",
        save_total_limit=1
    )

    # Initialize Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    if 'eval_f1' in eval_results:
        return eval_results['eval_f1']
    else:
        # Log an error or fallback to another metric
        print("F1 score not found in evaluation results. Using accuracy instead.")
        return eval_results['eval_accuracy']

if __name__ =="__main__":
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=1)

    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f)
    print(f"Best hyperparameters: {study.best_params}")