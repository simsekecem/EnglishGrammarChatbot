import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
import os
import logging

os.environ["WANDB_DISABLED"] = "true"
logging.disable(logging.WARNING)

df = pd.read_csv("Grammar Correction.csv")[["Ungrammatical Statement", "Standard English"]].dropna()
df["Ungrammatical Statement"] = df["Ungrammatical Statement"].str.lower().str.strip()
df["Standard English"] = df["Standard English"].str.lower().str.strip()
df = df.rename(columns={"Ungrammatical Statement": "input_text", "Standard English": "target_text"})
df["input_text"] = "fix grammar: " + df["input_text"]
df = df.sample(frac=1).reset_index(drop=True)

dataset = Dataset.from_pandas(df)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def preprocess(example):
    input_enc = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=128)
    target_enc = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(preprocess, batched=False)

training_args = TrainingArguments(
    output_dir="./t5-grammar-model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

# Eğitilmiş model ve tokenizer'ı kaydet
model.save_pretrained("./t5-grammar-model")
tokenizer.save_pretrained("./t5-grammar-model")
