from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Initialize the model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

def predict_sentiment(tweet):
    # Check if the tweet is a valid string and not empty
    if not isinstance(tweet, str) or not tweet.strip():
        return "neutral"  # Return neutral if input is invalid

    # Tokenize the tweet text
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()

    # Return the predicted sentiment label
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_map.get(predicted_class, 'neutral')  # Default to neutral if class not found
