# ingest_medquad.py
import os
import pandas as pd
from langchain.schema import Document
from sklearn.model_selection import train_test_split

DATA_CSV = "../data/medquad.csv"   # path to medQuad CSV
SPLIT_OUTPUT = "../data/medquad_splits"  # folder to save csv splits

os.makedirs(SPLIT_OUTPUT, exist_ok=True)

def load_medquad(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    # expect columns: question, answer, source, focus_area
    required = {"question","answer","source","focus_area"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns: {required}. Found: {df.columns.tolist()}")
    df = df.dropna(subset=["question","answer"]).reset_index(drop=True)
    return df

def df_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}\nFocus: {row.get('focus_area','')}"
        metadata = {"source": row.get("source",""), "focus_area": row.get("focus_area","")}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def create_train_eval_splits(df, test_size=0.1, random_state=42):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(os.path.join(SPLIT_OUTPUT, "train.csv"), index=False)
    test.to_csv(os.path.join(SPLIT_OUTPUT, "test.csv"), index=False)
    print(f"Saved train/test splits to {SPLIT_OUTPUT}")
    return train, test

if __name__ == "__main__":
    df = load_medquad()
    train, test = create_train_eval_splits(df)
    print("Sample row:", df.iloc[0].to_dict())
