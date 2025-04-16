from transformers import RobertaForSequenceClassification

def get_model(model_name="roberta-base", num_labels=3):
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
