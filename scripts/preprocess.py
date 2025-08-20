import os
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_FILE = "vox_machina_dnd_llm\data\fullparty_grouped.jsonl"


datasets = load_dataset("json", data_files=DATA_FILE)
train_test_split = datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])


train_dataset.save_to_disk("data/train_dataset")
eval_dataset.save_to_disk("data/eval_dataset")
