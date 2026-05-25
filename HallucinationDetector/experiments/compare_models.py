import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc

from src.data_loader import load_truthfulqa
from src.model_loader import load_model, generate_response_with_probs

MODELS = [
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen-3B"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),  
]


def simple_hallucination_label(response, correct_answer):
    resp_clean = response.lower().strip()
    correct_clean = correct_answer.lower().strip()
    if correct_clean in resp_clean:
        return 0
    else:
        return 1


def run_experiment_for_model(model_name, model_label, df, sample_size=20):
    print(f"\n{'=' * 60}")
    print(f"Запуск для модели: {model_label} ({model_name})")
    print('=' * 60)

    model, tokenizer = load_model(model_name, device_map="cpu")
    results = []

    for idx, row in df.iterrows():
        q = row['question']
        correct = row['best_answer']
        print(f"\n[{model_label}] Вопрос {idx + 1}/{len(df)}: {q[:80]}...")

        resp, prob = generate_response_with_probs(model, tokenizer, q, max_new_tokens=100)
        print(f"Ответ: {resp[:150]}...")
        print(f"Вероятность: {prob:.4f}")

        label = simple_hallucination_label(resp, correct)
        print(f"Автометка: {'галлюцинация' if label == 1 else 'правда'}")

        results.append({
            'model': model_label,
            'question': q,
            'model_response': resp,
            'correct_answer': correct,
            'avg_prob': prob,
            'auto_label': label,
            'detector_pred_07': 1 if prob < 0.7 else 0
        })

    df_result = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data" / "results" / f"{model_label}_auto.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в {output_path}")
    return df_result


def evaluate_and_plot(df_results, model_label):
    y_true = df_results['auto_label']
    y_pred = df_results['detector_pred_07']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Метрики для {model_label} (порог 0.7) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Truthful', 'Hallucination'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix (threshold=0.7) - {model_label}')
    plt.savefig(f'{model_label}_cm.png', dpi=150)
    plt.show()

    plt.figure()
    truthful = df_results[df_results['auto_label'] == 0]['avg_prob']
    hall = df_results[df_results['auto_label'] == 1]['avg_prob']
    plt.hist(truthful, bins=5, alpha=0.7, label='Truthful (auto)', color='green')
    plt.hist(hall, bins=5, alpha=0.7, label='Hallucination (auto)', color='red')
    plt.axvline(0.7, color='blue', linestyle='--', label='Threshold 0.7')
    plt.xlabel('Average token probability')
    plt.ylabel('Frequency')
    plt.title(f'Probability distribution - {model_label}')
    plt.legend()
    plt.savefig(f'{model_label}_hist.png', dpi=150)
    plt.show()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    N = 20  
    print(f"Загружаем {N} примеров из TruthfulQA...")
    df = load_truthfulqa(sample_size=N)

    all_metrics = {}
    for model_name, model_label in MODELS:
        df_res = run_experiment_for_model(model_name, model_label, df, sample_size=N)
        metrics = evaluate_and_plot(df_res, model_label)
        all_metrics[model_label] = metrics

    if len(all_metrics) > 1:
        models = list(all_metrics.keys())
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics_list))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model in enumerate(models):
            values = [all_metrics[model][m] for m in metrics_list]
            ax.bar(x + i * width, values, width, label=model)
        ax.set_ylabel('Score')
        ax.set_title('Comparison of models (auto-labeled)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics_list)
        ax.legend()
        plt.tight_layout()
        plt.savefig('model_comparison_auto.png', dpi=150)
        plt.show()

    print("\nГотово! Файлы сохранены: CSV, матрицы ошибок, гистограммы.")


if __name__ == "__main__":
    main()
