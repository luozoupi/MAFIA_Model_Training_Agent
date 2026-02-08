# finetune_lora.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model
import evaluate
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset (use subset for faster training)
    dataset = load_dataset("imdb")
    dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(2000))
    dataset["test"] = dataset["test"].shuffle(seed=args.seed).select(range(500))
    logger.info(f"Loaded IMDB dataset subset: {len(dataset['train'])} train, {len(dataset['test'])} test")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize function
    def tokenize_examples(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_examples, batched=True, remove_columns=["text"])

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer))

    # Configure LoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"]
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    logger.info("Starting training")
    train_result = trainer.train()
    logger.info(f"Training completed. Metrics: {train_result.metrics}")

    # Evaluate
    logger.info("Evaluating on test set")
    metrics = trainer.evaluate()
    logger.info(f"Final evaluation results: {metrics}")

    # Print final results clearly
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("="*60)

if __name__ == "__main__":
    main()
