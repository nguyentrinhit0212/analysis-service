import torch
from transformers import BertTokenizer, BertForSequenceClassification
class BertModel:
    """
    Class to handle topic classification using a pre-trained BERT model.
    """

    def __init__(self, model_name, tokenizer_name='bert-base-uncased'):
        """
        Initialize the model and tokenizer.
        
        Args:
            model_path (str): The path to the pre-trained model.
            tokenizer_name (str): The name of the pre-trained tokenizer (default: 'bert-base-uncased').
        """
        # Load the pre-trained model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained('./results/' + model_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Define the id2label mapping
        self.id2label = {
            "0": "Society & Culture",
            "1": "Science & Mathematics",
            "2": "Health",
            "3": "Education & Reference",
            "4": "Computers & Internet",
            "5": "Sports",
            "6": "Business & Finance",
            "7": "Entertainment & Music",
            "8": "Family & Relationships",
            "9": "Politics & Government"
        }

    def classify_topic(self, text, top_k=1):
        """
        Classify the topic of the given text and return the top-k topics.

        Args:
            text (str): The input text to classify.
            top_k (int): The number of top topics to return.

        Returns:
            list of tuples: A list of tuples containing the predicted topic labels and their scores.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get logits (predictions) from the model
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get the top-k predicted class IDs and their scores
        top_k_scores, top_k_class_ids = torch.topk(logits, top_k, dim=1)
        
        # Convert to list of tuples (topic label, score)
        top_k_topics = []
        for class_id, score in zip(top_k_class_ids[0], top_k_scores[0]):
            class_id_str = str(class_id.item())
            if class_id_str in self.id2label:
                top_k_topics.append((self.id2label[class_id_str], score.item()))

        return top_k_topics
