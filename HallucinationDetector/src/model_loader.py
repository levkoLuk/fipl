from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name, device_map="cpu", dtype=torch.float16):
    print(f"Загружаем модель: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response_with_probs(model, tokenizer, prompt, max_new_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.apply_chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\nAssistant: "

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            output_logits=True,
            return_dict_in_generate=True
        )

    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logits = outputs.logits
    probs = []
    for step, logit in enumerate(logits):
        step_logit = logit[0]  # (vocab_size,)
        step_prob = torch.softmax(step_logit, dim=-1)
        token_id = generated_ids[step]
        prob = step_prob[token_id].item()
        probs.append(prob)

    avg_prob = sum(probs) / len(probs) if probs else 0.0
    return response, avg_prob


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.apply_chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\nAssistant: "

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response
