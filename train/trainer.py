from transformers import Trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics):
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
