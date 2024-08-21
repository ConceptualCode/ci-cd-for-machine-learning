import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from evaluation import compute_metrics



def train_model():

    #check if CUDA is available
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available, Please make sure sure you have a GPU enabled environment")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatasetDict({
        'train': load_dataset('csv', data_files='preprocessed_data/train.csv')['train'],
        'validation': load_dataset('csv', data_files='preprocessed_data/val.csv')['train'],
        'test': load_dataset('csv', data_files='preprocessed_data/test.csv')['train']
    })

    num_labels = len(set(dataset['train']['label']))
    
    model_name = "checkpoint-17000"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.to(device)

    # tokenize data
    dataset = dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", truncation=True), batched=True)
    
    dataset = dataset.rename_column("label", "labels")


    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=1000,
    eval_steps=50,
    weight_decay=0.01,
    #logging_dir='./logs_nc',
    logging_strategy='steps',
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    warmup_steps=500,
    do_train=True,
    do_eval=True
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

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("models/fine_tuned_igbo_sentiment")
    tokenizer.save_pretrained("models/fine_tuned_igbo_sentiment")

if __name__ == "__main__":
    train_model()
