
def truncate_history(history, tokenizer, max_tokens=3000, system_prompt=""):
    while True:
        full_text = system_prompt + "\n" + "\n".join(history)
        tokenized = tokenizer(full_text, return_tensors="pt")
        if tokenized.input_ids.shape[1] <= max_tokens:
            break
        if len(history) > 1:
            history.pop(0)
        else:
            break
    return history
