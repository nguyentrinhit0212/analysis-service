import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./trained_model_yahoo_anwser_topics')  # Load the trained model

# Manually define the label mapping for Yahoo Answers Topics
id2label = {
    0: "Society & Culture",
    1: "Science & Mathematics",
    2: "Health",
    3: "Education & Reference",
    4: "Computers & Internet",
    5: "Sports",
    6: "Business & Finance",
    7: "Entertainment & Music",
    8: "Family & Relationships",
    9: "Politics & Government"
}

def classify_intent(text, top_k=3):
    """
    Classify the intent of the given text using the trained BERT model.
    
    Args:
        text (str): The input text to classify.
        top_k (int): The number of top intents to return.
    
    Returns:
        list of tuples: A list of tuples containing the predicted intent labels and their scores.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get logits (predictions) from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get the top-k predicted class IDs and their scores
    top_k_scores, top_k_class_ids = torch.topk(logits, top_k, dim=1)
    
    # Convert to list of tuples (intent label, score)
    top_k_intents = [
        (id2label[class_id.item()], score.item())  # Use manual id2label mapping
        for class_id, score in zip(top_k_class_ids[0], top_k_scores[0])
    ]
    
    return top_k_intents
