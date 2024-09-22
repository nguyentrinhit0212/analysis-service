import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from logger import logger

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available: {cuda_available}")

# Load the dataset
ds = load_dataset("community-datasets/yahoo_answers_topics", split="train")

# Combine relevant text fields and truncate if necessary to prevent exceeding the token limit
def combine_text_fields(examples):
    combined_text = (examples['question_title'] or "") + " " + (examples['question_content'] or "") + " " + (examples['best_answer'] or "")
    return {'text': combined_text.strip()}

# Log before starting the combine process
logger.info("Starting to combine text fields...")
ds = ds.map(combine_text_fields)
logger.info("Finished combining text fields.")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing function to tokenize
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Log before starting tokenization
logger.info("Starting tokenization of text...")
encoded_dataset = ds.map(preprocess_function, batched=True)
logger.info("Finished tokenization of text.")

# Add labels
def add_labels(examples):
    examples['labels'] = examples['topic']
    return examples

# Log before adding labels
logger.info("Starting to add labels...")
encoded_dataset = encoded_dataset.map(add_labels, batched=True)
logger.info("Finished adding labels.")

# Train/Test split (assuming no separate validation set)
train_test_split = encoded_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load pre-trained BERT model with the correct number of labels
num_labels = len(ds.features['topic'].names)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Move model to GPU if available
device = torch.device("cuda" if cuda_available else "cpu")
model.to(device)

# Use DataCollatorWithPadding to handle dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute evaluation metrics (Accuracy, F1, Precision, Recall)
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Training setup
def train_model():
    training_args = TrainingArguments(
        output_dir='./results/yahoo_answers',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Batch size per GPU
        per_device_eval_batch_size=8,   # Batch size per GPU
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,  # Mixed precision for faster training
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
        save_strategy="epoch",
        save_total_limit=2,  # Keep only the last 2 checkpoints
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Monitor accuracy for best model
        greater_is_better=True,
        report_to="tensorboard",  # Enable TensorBoard logging
        dataloader_num_workers=4,  # Set workers for faster data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # Use the custom metric function
    )

    trainer.train()

# Run the training process
if __name__ == "__main__":
    train_model()
