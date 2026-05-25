''''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_qwen(model_name="Qwen/Qwen2.5-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def generate_response_with_probs(model, tokenizer, prompt, max_new_tokens=100):

    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        step_logit = logit[0]
        step_prob = torch.softmax(step_logit, dim=-1)
        token_id = generated_ids[step]
        prob = step_prob[token_id].item()
        probs.append(prob)

    avg_prob = sum(probs) / len(probs) if probs else 0.0
    return response, avg_prob

if __name__ == "__main__":
    model, tokenizer = load_qwen()
    print(generate_response(model, tokenizer, "What is the capital of France?"))
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name, device_map="cpu", dtype=torch.float16):
    """
    Универсальная загрузка языковой модели и токенизатора.
    Поддерживает Qwen, Mistral, Llama и другие (с chat_template).
    """
    print(f"Загружаем модель: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # Для некоторых моделей (например, Mistral) нужно установить pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response_with_probs(model, tokenizer, prompt, max_new_tokens=100):
    """
    Генерирует ответ и возвращает (текст, средняя_вероятность_токенов)
    """
    messages = [{"role": "user", "content": prompt}]
    # Применяем chat_template, если он есть, иначе простой формат
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
    """
    Упрощённая генерация ответа (без возврата вероятностей).
    Используется для SelfCheckGPT.
    """
    messages = [{"role": "user", "content": prompt}]
    # Пытаемся использовать chat_template, если он есть
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
    # Декодируем только новые токены
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response