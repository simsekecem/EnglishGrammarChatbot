# English Grammar Chatbot (T5-base)

This project implements a sequence-to-sequence NLP pipeline to identify and correct English grammatical errors. It leverages a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model to transform "ungrammatical" input text into "standard" English.

## Project Architecture & File Roles

The project is organized into modular scripts handling data preparation, model training, and user interface deployment.

### 1. Data Foundation: `Grammar Correction.csv`
The core of the model is this dataset containing over 2,000 sentence pairs. It consists of:
- **Ungrammatical Statement:** The raw input with errors (e.g., *"I goes to the store"*).
- **Standard English:** The target ground truth (e.g., *"I go to the store"*).
- **Error Type:** Categorical labels like Verb Tense, Subject-Verb Agreement, and Prepositions.

### 2. Training Logic: `train_model.py`
This script handles the fine-tuning process of the `t5-base` model.
- **Preprocessing:** It prepends the prefix `"fix grammar: "` to all inputs, consistent with T5's multi-task architecture.
- **Tokenization:** Uses the T5Tokenizer with a max length of 128 tokens.
- **Optimization:** Utilizes the Hugging Face `Trainer` API with specific hyperparameters (batch size of 4, 3 epochs).
- **Output:** Saves the final weights and tokenizer configuration to the `./t5-grammar-model` directory.

### 3. Deployment & UI: `run_model.py` & `run_model.bat`
This is the entry point for using the system after training.
- **Inference Pipeline:** Loads the local model from `./t5-grammar-model`, moves it to GPU (if available), and uses beam search (4 beams) for high-quality text generation.
- **Gradio Interface:** Provides a web-based GUI where users can type sentences and see corrections instantly.
- **Batch Execution:** `run_model.bat` allows for one-click launching on Windows environments.

### 4. Combined Script: `grammar_ui.py`
A standalone version that combines both training and UI launching logic. It is useful for end-to-end testing or for environments where training and inference happen in the same session.

---

## Technical Workflow

1. **Dataset Loading:** Scripts read `Grammar Correction.csv` and clean the text (lowercase/strip).
2. **Fine-Tuning:** `train_model.py` updates the T5 weights to specialize in grammar correction.
3. **Model Persistence:** The trained state is stored in `t5-grammar-model/`.
4. **Interaction:** `run_model.py` provides the interface to query the trained model.

## Requirements
- **Hardware:** GPU recommended (CUDA support is automatically detected).
- **Libraries:** `transformers`, `torch`, `gradio`, `pandas`, `datasets`.

