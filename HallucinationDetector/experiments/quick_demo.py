import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.data_loader import load_truthfulqa
from src.model_loader import load_qwen, generate_response_with_probs

def main():
    df = load_truthfulqa(sample_size=5)
    model, tokenizer = load_qwen()
    results = []
    for idx, row in df.iterrows():
        q = row['question']
        correct = row['best_answer']
        resp, prob = generate_response_with_probs(model, tokenizer, q, max_new_tokens=100)
        results.append({
            'question': q,
            'response': resp,
            'correct': correct,
            'avg_prob': prob,
            'hallucination_guess': prob < 0.7
        })
        print(f"Prob: {prob:.3f} | Hall? {prob < 0.7} | {q[:50]}...")
    pd.DataFrame(results).to_csv('demo_results.csv', index=False)
    print("Saved demo_results.csv")

if __name__ == "__main__":
    main()