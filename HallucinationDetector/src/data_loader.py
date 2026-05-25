from datasets import load_dataset
import pandas as pd

def load_truthfulqa(split="validation", sample_size=None):

    print(f"Загружаем TruthfulQA (split={split})...")
    dataset = load_dataset("truthful_qa", "generation", split=split)
    df = pd.DataFrame(dataset)
    if sample_size is not None:
        df = df.head(sample_size)
    print(f"Загружено {len(df)} примеров")
    return df

if __name__ == "__main__":
    df = load_truthfulqa(sample_size=5)
    print(df[['question', 'best_answer']].head())