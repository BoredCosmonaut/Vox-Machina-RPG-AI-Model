# Vox Machina RPG AI Model

This repository contains a fine-tuned language model designed to simulate a single-player RPG experience with **Vox Machina** from *Critical Role*. The model generates immersive dialogue for the player, the Dungeon Master (DM), and party members.

---

## Project Overview

- **Purpose:** Create an AI-powered RPG experience where the player interacts with Vox Machina and a DM in a text-based chat.
- **Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Fine-tuning:** Using LoRA on a dataset of structured prompts and responses inspired by *Critical Role* episodes.

> **Note:** This model is for personal or research use only. All characters, settings, and story elements are based on *Critical Role*, created by Matthew Mercer and the cast.

---

## Repository Structure
```

├── data/ # Raw and preprocessed datasets (JSONL)
├── scripts/ # Training and preprocessing scripts
├── model/ # Fine-tuned model weights and tokenizer
├── inference/ # Chat function and CLI game loop
├── utils/ # utility functions
└── README.md

```
## Dataset

- Format: JSONL with `prompt` and `response` fields.  
- Example:
```json
{
    "prompt":"SAM: Adra! You run a fantastic establishment.",
    "response":"DM: “That's right I do!”"
}
```
## Credits

- **Critical Role / Matthew Mercer and Cast**  
  All characters, storylines, and original content belong to *Critical Role Productions, LLC*. This project is **not officially affiliated** or endorsed.

- **Dataset Sources**  
  Derived from publicly available Critical Role <a href="https://criticalrole.fandom.com/wiki/Category:Transcripts">transcripts</a>.

- **Libraries & Tools**  
  - [Vue.js](https://vuejs.org/) for frontend  
  - [Pinia](https://pinia.vuejs.org/) for state management  
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model training and inference  
  - [LoRA / PEFT](https://github.com/huggingface/peft) for fine-tuning  

> ⚠️ This project is for **personal, educational, or research purposes only**.
