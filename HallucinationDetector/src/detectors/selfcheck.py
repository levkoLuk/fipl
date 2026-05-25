from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import spacy
import torch

# Загружаем модель для разбивки текста на предложения
# Важно: используем блок `if`, чтобы загружать её только один раз
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Модель spaCy не найдена. Загружаем...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def run_selfcheck_consistency(model, tokenizer, prompt, original_response, num_samples=3):
    """
    Оценивает согласованность ответа модели с помощью SelfCheckGPT.
    Возвращает: (consistency_score, error)
    - consistency_score: float от 0 до 1, где 0 = абсолютно согласован, 1 = полная галлюцинация
    - error: текст ошибки или None
    """
    try:
        # 1. Генерируем альтернативные ответы
        print(f"    -> Генерация {num_samples} альтернативных ответов для SelfCheck...")
        alternative_responses = []
        for i in range(num_samples):
            alt_response = generate_response(model, tokenizer, prompt)
            alternative_responses.append(alt_response)
            print(f"        Сгенерирован ответ {i+1}/{num_samples}")

        # 2. Разбиваем оригинальный ответ на предложения
        sentences = [sent.text.strip() for sent in nlp(original_response).sents]
        if not sentences:
            # На случай, если ответ пустой или очень короткий
            sentences = [original_response]

        # 3. Инициализируем SelfCheck с BERTScore
        # Устанавливаем `rescale_with_baseline=True`, чтобы оценки были в диапазоне от 0 до 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selfcheck = SelfCheckBERTScore(rescale_with_baseline=True, device=device)

        # 4. Вычисляем оценку согласованности для каждого предложения
        # Функция вернёт список оценок (например, [0.1, 0.2, 0.9]) для каждого предложения.
        # Высокое значение означает, что предложение сильно противоречит альтернативным ответам = галлюцинация.
        sent_scores = selfcheck.predict(
            sentences=sentences,                     # Список предложений для проверки
            sampled_passages=alternative_responses   # Альтернативные ответы для сравнения
        )

        # 5. Агрегируем оценку для всего ответа: берём максимум
        # Если хотя бы одно предложение — галлюцинация, считаем весь ответ проблемным.
        consistency_score = max(sent_scores) if sent_scores else 0.0
        return consistency_score, None

    except Exception as e:
        print(f"    ! Ошибка в SelfCheckGPT: {e}")
        return 0.0, str(e)

# Импортируем generate_response для использования внутри selfcheck
# Это нужно в конце, чтобы избежать циклических импортов
from src.model_loader import generate_response