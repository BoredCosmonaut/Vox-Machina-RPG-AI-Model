
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from utils.system_prompt import build_system_prompt, current_party
from utils.history import truncate_history


model_path = "vox_machina_dnd_llm\model\mistral-7B-DM"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

conversation_history = []
system_prompt = build_system_prompt(current_party)


class StopOnSpeaker(StoppingCriteria):
    def __init__(self, tokenizer, speakers):
        self.stop_ids = [tokenizer(speaker).input_ids for speaker in speakers]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids.shape[1] < len(stop_id):
                continue
            if (input_ids[0, -len(stop_id):] == torch.tensor(stop_id).to(input_ids.device)).all():
                return True
        return False

stop_criteria = StoppingCriteriaList([StopOnSpeaker(tokenizer, ["Player:"] + [f"{name}:" for name in current_party] + ["DM:"])])

def chat(player_input, max_new_tokens=500):
    global conversation_history
    conversation_history.append(f"Player: {player_input}")
    conversation_history = truncate_history(conversation_history, tokenizer, system_prompt=system_prompt)
    full_prompt = system_prompt + "\n" + "\n".join(conversation_history) + "\n"

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_output[len(full_prompt):].strip()
    conversation_history.append(response)
    return response
