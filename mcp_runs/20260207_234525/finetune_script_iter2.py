import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import evaluate
import torch
import logging

# Configure logging to capture all messages
logging.basicConfig(level=logging.INFO)

# Load IMDB dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Split train into train/val
train_val = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_val["train"]
val_dataset = train_val["test"]

# Load tokenizer and model without quantization first
model_id = "google/gemma-2-2b-it"

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Setup PEFT configuration with adjusted target modules for Gemma
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["self_attn.q_proj", "self_attn.v_proj"],  # Adjusted for Gemma architecture
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Define metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"]
    }

# Training arguments with reduced batch size
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced from 16 to 8
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train with full error propagation
print("Starting training...")
try:
    train_result = trainer.train()
    print(f"Training completed. Metrics: {train_result.metrics}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized_val)
    print(f"Test metrics: {test_metrics}")
    
    # Save model
    trainer.save_model("./final_model")
except Exception as e:
    logging.exception("Training failed with error:")
    raise
