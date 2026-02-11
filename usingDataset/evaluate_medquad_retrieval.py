# evaluate_medquad_retrieval.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest_medquad import load_medquad, create_train_eval_splits

# Config
SPLIT_DIR = "../data/medquad_splits"
SEMANTIC_EVAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # consistent eval model
THRESHOLD = 0.45  # semantic match threshold
K = 5

# Embedding models to test (use your mapping)
embedding_models = {
    "BioBERT": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
}

def semantic_match(keyword, text, threshold=THRESHOLD):
    if not text or not keyword:
        return False
    emb_k = SEMANTIC_EVAL_MODEL.encode(keyword, convert_to_tensor=True)
    emb_t = SEMANTIC_EVAL_MODEL.encode(text, convert_to_tensor=True)
    sim = util.cos_sim(emb_k, emb_t).item()
    return sim >= threshold

def evaluate_db(db, test_df, k=K):
    y_true, y_pred = [], []
    test_df = test_df.fillna("")
    per_query_f1 = []
    for _, row in test_df.iterrows():
        q = row["question"]
        keywords = [w.strip().lower() for w in (row["answer"] + " " + row.get("focus_area","")).split() if len(w)>2][:6]
        # Retrieved from db
        results = db.similarity_search(q, k=k)
        retrieved = " ".join([r.page_content for r in results]).lower()

        q_true = []
        q_pred = []
        for kw in keywords:
            q_true.append(1)
            matched = semantic_match(kw, retrieved)
            q_pred.append(1 if matched else 0)

        y_true.extend(q_true)
        y_pred.extend(q_pred)
        if any(q_pred):
            per_query_f1.append(f1_score(q_true, q_pred, zero_division=0))
        else:
            per_query_f1.append(0.0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mean_f1 = float(np.mean(per_query_f1)) if per_query_f1 else 0.0
    return precision, recall, f1, accuracy, mean_f1

if __name__ == "__main__":
    # load splits (make sure ingest_medquad created them)
    test_df = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    # Example: evaluate the FAISS you built earlier (sequential)
    from build_faiss_from_medquad import DB_OUTPUT, EMBEDDING_MODEL
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(DB_OUTPUT, emb, allow_dangerous_deserialization=True)
    p, r, f1, acc, mean_f1 = evaluate_db(db, test_df)
    print("Sequential FAISS (single model) ->", {"precision":p,"recall":r,"f1":f1,"acc":acc,"mean_f1":mean_f1})
