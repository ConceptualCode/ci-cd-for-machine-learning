import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from evaluation import compute_metrics
import json
import matplotlib.pyplot as plt


def save_metrics(trainer, output_dir="./results"):
    metrics = trainer.evaluate()
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(str(metrics))
    
    # Example plot (if applicable)
    plt.plot(metrics['eval_loss'])
    plt.savefig(f"{output_dir}/plots.png")


import matplotlib.pyplot as plt

def plot_metrics(trainer):
    # Assuming 'trainer' has a history of metrics across epochs
    metrics_history = trainer.state.log_history  # This is a list of dicts
    
    eval_losses = [entry['eval_loss'] for entry in metrics_history if 'eval_loss' in entry]
    
    if eval_losses:
        epochs = range(1, len(eval_losses) + 1)
        plt.plot(epochs, eval_losses, 'bo-', label='Evaluation loss')
        plt.title('Evaluation loss across epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./results/plots.png')
        plt.close()
    else:
        print("No evaluation losses found in the training logs.")


def train_model():
    # Load best hyperparameters from file
    with open("best_hyperparameters.json", "r") as f:
        best_params = json.load(f)

    # #check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available")
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatasetDict({
        'train': load_dataset('csv', data_files='preprocessed_data/train.csv')['train'],
        'validation': load_dataset('csv', data_files='preprocessed_data/val.csv')['train'],
        'test': load_dataset('csv', data_files='preprocessed_data/test.csv')['train']
    })

    num_labels = len(set(dataset['train']['label']))
    
    model_name = "checkpoint-1520000"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.to(device)

    # tokenize data
    dataset = dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", max_length=512, truncation=True), batched=True)

    dataset = dataset.rename_column("label", "labels")

    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=best_params['num_train_epochs'],
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=16, #best_params['per_device_batch_size'],
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    save_metrics(trainer)
    plot_metrics(trainer)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("models/fine_tuned_igbo_sentiment")
    tokenizer.save_pretrained("models/fine_tuned_igbo_sentiment")

if __name__ == "__main__":
    train_model()
