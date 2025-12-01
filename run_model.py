import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

# Modeli yükle
model_path = "./t5-grammar-model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct_grammar(sentence):
    sentence = sentence.lower().strip()
    input_text = "fix grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

interface = gr.Interface(
    fn=correct_grammar,
    inputs=gr.Textbox(label="Yanlış cümle"),
    outputs=gr.Textbox(label="Düzeltilmiş cümle"),
    title="Grammar Corrector (T5)",
    description="Bir cümle girin, model gramerini düzeltsin."
)

interface.launch()
