import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_truthfulqa
from src.model_loader import load_model, generate_response_with_probs
from src.detectors.selfcheck import run_selfcheck_consistency

def simple_hallucination_label(response, correct_answer):
    if correct_answer.lower() in response.lower():
        return 0
    else:
        return 1

def main():
    N = 10 
    print(f"Загружаем {N} примеров из TruthfulQA...")
    df = load_truthfulqa(sample_size=N)

    print("Загружаем модель Qwen-3B...")
    model, tokenizer = load_model("Qwen/Qwen2.5-3B-Instruct", device_map="cpu")
    print("Модель загружена.")

    results = []
    for idx, row in df.iterrows():
        question = row['question']
        correct_answer = row['best_answer']

        print(f"\n--- Обработка вопроса {idx+1}/{N} ---")
        print(f"Q: {question[:100]}...")

        response, avg_prob = generate_response_with_probs(model, tokenizer, question)
        print(f"Ответ модели: {response[:150]}...")
        print(f"Средняя вероятность токенов (Token Probability): {avg_prob:.4f}")

        selfcheck_score, selfcheck_error = run_selfcheck_consistency(model, tokenizer, question, response, num_samples=3)
        if selfcheck_error:
            print(f"    SelfCheckGPT завершился с ошибкой: {selfcheck_error}")
        else:
            print(f"Оценка SelfCheckGPT: {selfcheck_score:.4f} (0=согласован, 1=галлюцинация)")

        label = simple_hallucination_label(response, correct_answer)
        print(f"Автометка (truthful=0, hallucination=1): {label}")

        results.append({
            'question': question,
            'model_response': response,
            'correct_answer': correct_answer,
            'token_prob': avg_prob,
            'selfcheck_score': selfcheck_score if not selfcheck_error else None,
            'auto_label': label,
        })

    df_results = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data" / "results" / "selfcheck_qwen_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в {output_path}")

    df_results['pred_token'] = (df_results['token_prob'] < 0.7).astype(int)
    df_results['pred_selfcheck'] = (df_results['selfcheck_score'] > 0.5).astype(int)
    y_true = df_results['auto_label']

    acc_token = accuracy_score(y_true, df_results['pred_token'])
    prec_token = precision_score(y_true, df_results['pred_token'], zero_division=0)
    rec_token = recall_score(y_true, df_results['pred_token'], zero_division=0)
    f1_token = f1_score(y_true, df_results['pred_token'], zero_division=0)

    acc_sc = accuracy_score(y_true, df_results['pred_selfcheck'])
    prec_sc = precision_score(y_true, df_results['pred_selfcheck'], zero_division=0)
    rec_sc = recall_score(y_true, df_results['pred_selfcheck'], zero_division=0)
    f1_sc = f1_score(y_true, df_results['pred_selfcheck'], zero_division=0)

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*50)
    print(f"{'Метрика':<15} {'Token Probability':<25} {'SelfCheckGPT':<25}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {acc_token:<25.4f} {acc_sc:<25.4f}")
    print(f"{'Precision':<15} {prec_token:<25.4f} {prec_sc:<25.4f}")
    print(f"{'Recall':<15} {rec_token:<25.4f} {rec_sc:<25.4f}")
    print(f"{'F1-Score':<15} {f1_token:<25.4f} {f1_sc:<25.4f}")

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Token Probability': [acc_token, prec_token, rec_token, f1_token],
        'SelfCheckGPT': [acc_sc, prec_sc, rec_sc, f1_sc]
    })
    metrics_df.plot(x='Metric', kind='bar', figsize=(10, 6))
    plt.title('Сравнение методов детекции галлюцинаций')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('selfcheck_comparison.png')
    plt.show()

    print("\nГрафик сохранён как 'selfcheck_comparison.png'")

if __name__ == "__main__":
    main()
