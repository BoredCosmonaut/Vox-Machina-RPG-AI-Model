# scripts/train.py
import math
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk


train_dataset = load_from_disk("data/train_dataset")
eval_dataset = load_from_disk("data/eval_dataset")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=10,
    num_train_epochs=30,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-4,
    bf16=True,
    report_to="none",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


trainer.train()


eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"Eval perplexity: {perplexity:.3f}")


trainer.save_model("vox_machina_dnd_llm\model\mistral-7B-DM")
tokenizer.save_pretrained("vox_machina_dnd_llm\model\mistral-7B-DM")
