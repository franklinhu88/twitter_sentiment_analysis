from datasets import load_dataset
from transformers import RobertaTokenizer

def load_and_tokenize_dataset(tokenizer_name="roberta-base"):
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    
    return tokenized["train"], tokenized["validation"], tokenized["test"], tokenizer
