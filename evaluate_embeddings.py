import os
import re
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Evaluation
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# NLP utils
import nltk
import spacy
from dotenv import load_dotenv

nltk.download('punkt', quiet=True)

load_dotenv()

# Config
DATA_PATH = "data/"                 
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
CHUNK_LIMIT = 250                         # limit chunks per model for speed
K = 5                                     # top-k retrieval
SEMANTIC_EVAL_MODEL = "all-MiniLM-L6-v2"  # used for semantic_match evaluation (keeps evaluation consistent)

# Embedding models to be compared
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BioBERT": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5"
}

# Extended test queries (adapt / expand for your dataset)
test_queries = [
    # Drug-related
    ("What are the side effects of metformin?", ["metformin", "side effects", "diabetes"]),
    ("How should amoxicillin be taken for bacterial infection?", ["amoxicillin", "infection", "dosage"]),
    ("What are the uses of paracetamol?", ["paracetamol", "uses", "fever", "pain"]),
    ("Can ibuprofen cause stomach ulcers?", ["ibuprofen", "ulcer", "pain relief", "NSAID"]),

    # Disease symptoms
    ("Symptoms of chronic kidney disease?", ["kidney", "symptoms", "CKD"]),
    ("Early signs of diabetes mellitus?", ["diabetes", "symptoms", "sugar", "insulin"]),
    ("What are the symptoms of pneumonia?", ["pneumonia", "symptoms", "cough", "fever"]),
    ("Indications of liver failure?", ["liver", "failure", "jaundice", "bilirubin"]),

    # Treatment / management
    ("Treatment options for hypertension?", ["hypertension", "treatment", "blood pressure"]),
    ("How is asthma managed?", ["asthma", "inhaler", "bronchodilator", "treatment"]),
    ("What are the medications used for depression?", ["depression", "SSRI", "antidepressant"]),
    ("How to treat bacterial meningitis?", ["meningitis", "bacterial", "antibiotic", "infection"]),

    # Anatomy / physiology
    ("How does insulin regulate blood sugar?", ["insulin", "glucose", "blood sugar", "pancreas"]),
    ("Function of red blood cells?", ["RBC", "oxygen", "hemoglobin"]),
    ("What does the liver do in metabolism?", ["liver", "metabolism", "detoxification"]),
    ("Role of kidneys in filtration?", ["kidney", "filtration", "urine", "nephrons"]),

    # Pathology / diagnostics
    ("What test confirms anemia?", ["anemia", "hemoglobin", "blood test"]),
    ("How is COVID-19 diagnosed?", ["COVID", "PCR", "swab", "virus"]),
    ("What causes high cholesterol levels?", ["cholesterol", "LDL", "diet", "lipid"]),
    ("How to interpret ECG abnormalities?", ["ECG", "heart", "rhythm", "arrhythmia"]),
]

# Utility functions
def load_pdf_pages(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs in {data_path}")
    return docs

def create_text_chunks(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks

# Semantic-match function (consistent across models)
semantic_model = SentenceTransformer(SEMANTIC_EVAL_MODEL)

def semantic_match(keyword: str, text: str, threshold: float = 0.45) -> bool:
    """
    Returns True if semantic similarity between keyword and text >= threshold.
    Using a consistent (fast) model for evaluation so comparisons are fair.
    """
    if not text or not keyword:
        return False
    emb_kw = semantic_model.encode(keyword, convert_to_tensor=True)
    emb_text = semantic_model.encode(text, convert_to_tensor=True)
    sim = util.cos_sim(emb_kw, emb_text).item()
    return sim >= threshold

def evaluate_faiss(db: FAISS, queries=test_queries, k=K, semantic=True):
    """Evaluate a built FAISS index using the provided queries and semantic matching."""
    y_true, y_pred = [], []
    per_query_f1 = []

    for q, keywords in queries:
        results = db.similarity_search(q, k=k)
        retrieved_texts = " ".join([r.page_content.lower() for r in results])

        query_true, query_pred = [], []
        for kw in keywords:
            query_true.append(1)
            if semantic:
                matched = semantic_match(kw.lower(), retrieved_texts)
            else:
                matched = kw.lower() in retrieved_texts
            query_pred.append(1 if matched else 0)

        y_true.extend(query_true)
        y_pred.extend(query_pred)

        # per-query F1 (binary)
        if any(query_pred):
            f1_q = f1_score(query_true, query_pred, zero_division=0)
        else:
            f1_q = 0.0
        per_query_f1.append(f1_q)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mean_f1 = float(np.mean(per_query_f1)) if per_query_f1 else 0.0

    return precision, recall, f1, accuracy, mean_f1

# Main experiment loop
def run_experiments():
    # Load & chunk once (shared)
    documents = load_pdf_pages(DATA_PATH)
    chunks = create_text_chunks(documents)

    if CHUNK_LIMIT is not None:
        print(f"Limiting to first {CHUNK_LIMIT} chunks for speed.")
        chunks_to_use = chunks[:CHUNK_LIMIT]
    else:
        chunks_to_use = chunks

    results = []
    os.makedirs("eval_results", exist_ok=True)

    for model_name, model_id in embedding_models.items():
        print(f"\n--- Evaluating embedding model: {model_name} ({model_id}) ---")

        try:
            # Initialize embedding wrapper
            emb = HuggingFaceEmbeddings(model_name=model_id)

            # Build FAISS. Use processed chunk Documents directly.
            docs_for_index = [Document(page_content=c.page_content) for c in chunks_to_use]
            db = FAISS.from_documents(docs_for_index, emb)

            # Evaluate
            pr, rc, f1, acc, mean_f1 = evaluate_faiss(db, queries=test_queries, k=K, semantic=True)
            print(f"{model_name} -> Precision: {pr:.3f}, Recall: {rc:.3f}, F1: {f1:.3f}, Acc: {acc:.3f}, MeanF1: {mean_f1:.3f}")

            results.append({
                "model": model_name,
                "hf_id": model_id,
                "precision": pr,
                "recall": rc,
                "f1": f1,
                "accuracy": acc,
                "mean_f1_query": mean_f1
            })

            del db

        except Exception as e:
            print(f"Error with model {model_name} ({model_id}): {e}")
            results.append({
                "model": model_name,
                "hf_id": model_id,
                "precision": None,
                "recall": None,
                "f1": None,
                "accuracy": None,
                "mean_f1_query": None,
                "error": str(e)
            })

    # Save results CSV
    csv_path = os.path.join("eval_results", "embedding_model_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["model"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved results -> {csv_path}")

    # Plot results (Precision/Recall/F1)
    good = [r for r in results if r.get("f1") is not None]
    if good:
        models = [r["model"] for r in good]
        precisions = [r["precision"] for r in good]
        recalls = [r["recall"] for r in good]
        f1s = [r["f1"] for r in good]
        accuracies = [r["accuracy"] for r in good]

        x = np.arange(len(models))
        width = 0.18

        plt.figure(figsize=(12,6))
        plt.bar(x - 1.5*width, precisions, width, label="Precision")
        plt.bar(x - 0.5*width, recalls, width, label="Recall")
        plt.bar(x + 0.5*width, f1s, width, label="F1")
        plt.bar(x + 1.5*width, accuracies, width, label="Accuracy")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.ylim(0,1)
        plt.ylabel("Score")
        plt.title("Embedding Model Comparison (Precision / Recall / F1 / Accuracy)")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join("eval_results", "embedding_model_comparison.png")
        plt.savefig(plot_path)
        print(f"Saved comparison plot -> {plot_path}")
        plt.show()
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    run_experiments()
