import os
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.utils.logger import AppLogger

class BertTrainer:
    def __init__(self, model_name='bert-base-uncased', dataset_name="community-datasets/yahoo_answers_topics", preprocessed_dataset_path='./preprocessed_dataset'):
        """
        Initialize the BertTrainer with the model, tokenizer, dataset, and path to save preprocessed data.

        Args:
            model_name (str): The name of the pretrained BERT model to use.
            dataset_name (str): The name of the dataset to load from HuggingFace Datasets.
            preprocessed_dataset_path (str): The path to save and load preprocessed dataset.
        """
        self.logger = AppLogger().get_logger()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.preprocessed_dataset_path = preprocessed_dataset_path

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Load or preprocess the dataset
        self.encoded_dataset = self.load_or_preprocess_dataset()

        # Train/Test split
        self.train_dataset, self.eval_dataset = self.split_dataset()

        # Data Collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def load_or_preprocess_dataset(self):
        """
        Load the dataset and either preprocess it or load it from disk.

        Returns:
            Dataset: Preprocessed dataset ready for training.
        """
        if os.path.exists(self.preprocessed_dataset_path):
            self.logger.info("Loading preprocessed dataset from disk...")
            return load_from_disk(self.preprocessed_dataset_path)
        else:
            self.logger.info("Loading dataset from HuggingFace...")
            ds = load_dataset(self.dataset_name, split="train")
            self.logger.info("Starting to combine text fields...")
            ds = ds.map(self.combine_text_fields)

            self.logger.info("Starting tokenization of text...")
            encoded_dataset = ds.map(self.preprocess_function, batched=True)

            self.logger.info("Starting to add labels...")
            encoded_dataset = encoded_dataset.map(self.add_labels, batched=True)

            self.logger.info("Saving preprocessed dataset to disk...")
            encoded_dataset.save_to_disk(self.preprocessed_dataset_path)
            return encoded_dataset

    def combine_text_fields(self, examples):
        """
        Combine relevant text fields from the dataset.
        """
        combined_text = (examples['question_title'] or "") + " " + (examples['question_content'] or "") + " " + (examples['best_answer'] or "")
        return {'text': combined_text.strip()}

    def preprocess_function(self, examples):
        """
        Tokenize the input text using the BERT tokenizer.
        """
        return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    def add_labels(self, examples):
        """
        Add the labels for classification from the dataset.
        """
        examples['labels'] = examples['topic']
        return examples

    def split_dataset(self):
        """
        Split the preprocessed dataset into train and test sets.

        Returns:
            tuple: Train and test datasets.
        """
        train_test_split = self.encoded_dataset.train_test_split(test_size=0.1)
        return train_test_split['train'], train_test_split['test']

    def train_model(self, output_dir='./results/yahoo_answers', num_epochs=1, learning_rate=3e-5):
        """
        Train the BERT model using the preprocessed dataset.

        Args:
            output_dir (str): Directory to save the trained model and logs.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="tensorboard",
            dataloader_num_workers=8,
            fp16=True,
            gradient_accumulation_steps=2,
        )

        trainer = Trainer(
            model=BertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.encoded_dataset.features['topic'].names)),
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model('./trained_model_yahoo_answers')


# Example usage
if __name__ == "__main__":
    bert_trainer = BertTrainer()
    bert_trainer.train_model()