import torch

def get_token_probabilities(model, tokenizer, prompt, response):

    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = full_prompt + response

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, :-1, :]

    probs = torch.softmax(logits, dim=-1)

    token_probs = []
    for i, input_id in enumerate(inputs.input_ids[0][1:], start=0):
        prob = probs[i, input_id].item()
        token_probs.append(prob)

    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return avg_prob, token_probs


def classify_by_token_prob(avg_prob, threshold=0.9):
    return "truthful" if avg_prob > threshold else "hallucination"