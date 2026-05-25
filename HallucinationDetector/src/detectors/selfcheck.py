from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import spacy
import torch
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Модель spaCy не найдена. Загружаем...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def run_selfcheck_consistency(model, tokenizer, prompt, original_response, num_samples=3):
    try:
        print(f"    -> Генерация {num_samples} альтернативных ответов для SelfCheck...")
        alternative_responses = []
        for i in range(num_samples):
            alt_response = generate_response(model, tokenizer, prompt)
            alternative_responses.append(alt_response)
            print(f"        Сгенерирован ответ {i+1}/{num_samples}")
        sentences = [sent.text.strip() for sent in nlp(original_response).sents]
        if not sentences:
            sentences = [original_response]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selfcheck = SelfCheckBERTScore(rescale_with_baseline=True, device=device)
        sent_scores = selfcheck.predict(
            sentences=sentences,                     
            sampled_passages=alternative_responses   
        )
        consistency_score = max(sent_scores) if sent_scores else 0.0
        return consistency_score, None

    except Exception as e:
        print(f"    ! Ошибка в SelfCheckGPT: {e}")
        return 0.0, str(e)

from src.model_loader import generate_response
