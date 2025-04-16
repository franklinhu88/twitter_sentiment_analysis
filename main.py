from data.load_dataset import load_and_tokenize_dataset
from models.roberta_model import get_model
from utils.metrics import compute_metrics
from config.training_args import get_training_args
from train.trainer import train_model

def main():
    train_ds, val_ds, test_ds, tokenizer = load_and_tokenize_dataset()
    model = get_model()
    training_args = get_training_args()
    trainer = train_model(model, tokenizer, train_ds, val_ds, training_args, compute_metrics)
    trainer.save_model("./roberta-sentiment")
    tokenizer.save_pretrained("./roberta-sentiment")

if __name__ == "__main__":
    main()
