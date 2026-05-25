import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_truthfulqa
from src.model_loader import load_qwen, generate_response
from tqdm import tqdm


def main():
    print("Загружаем TruthfulQA (split=validation)...")
    df = load_truthfulqa(sample_size=20)
    print(f"Загружено {len(df)} примеров")

    print("Загружаем модель Qwen/Qwen2.5-3B-Instruct...")
    model, tokenizer = load_qwen()
    print("Модель загружена")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка"):
        question = row['question']
        correct_answer = row['best_answer']

        response = generate_response(model, tokenizer, question)

        results.append({
            'question': question,
            'model_response': response,
            'correct_answer': correct_answer
        })

    import pandas as pd
    output_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data" / "results" / "baseline_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Результаты сохранены в {output_path}")

    print("\nПримеры:")
    for i in range(min(3, len(results))):
        print(f"\nВопрос: {results[i]['question']}")
        print(f"Ответ модели: {results[i]['model_response'][:200]}...")
        print(f"Правильный ответ: {results[i]['correct_answer']}")


if __name__ == "__main__":
    main()