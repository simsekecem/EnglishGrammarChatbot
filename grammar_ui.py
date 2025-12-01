import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
import gradio as gr
import os
import logging

# Ortam ayarları
os.environ["WANDB_DISABLED"] = "true"
logging.disable(logging.WARNING)

# Veriyi yükle
df = pd.read_csv("Grammar Correction.csv")  # CSV dosyan aynı dizinde olmalı
df = df[["Ungrammatical Statement", "Standard English"]].dropna()
df["Ungrammatical Statement"] = df["Ungrammatical Statement"].str.lower().str.strip()
df["Standard English"] = df["Standard English"].str.lower().str.strip()
df = df.rename(columns={"Ungrammatical Statement": "input_text", "Standard English": "target_text"})
df["input_text"] = "fix grammar: " + df["input_text"]
df = df.sample(frac=1).reset_index(drop=True)

# HuggingFace dataset formatına çevir
dataset = Dataset.from_pandas(df)

# Tokenizer ve model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Ön işleme
def preprocess(example):
    input_enc = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=128)
    target_enc = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(preprocess, batched=False)

# Eğitim ayarları
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

# Model eğitimi
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
trainer.train()

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tahmin fonksiyonu
def correct_grammar(sentence):
    sentence = sentence.lower().strip()
    input_text = "fix grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Gradio arayüzü
interface = gr.Interface(
    fn=correct_grammar,
    inputs=gr.Textbox(label="Yanlış cümle"),
    outputs=gr.Textbox(label="Düzeltilmiş cümle"),
    title="Grammar Corrector (T5)",
    description="Bir cümle girin, model gramerini düzeltsin."
)

interface.launch()
