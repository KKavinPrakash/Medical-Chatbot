import os, re, string
import nltk, spacy
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# SETUP
load_dotenv()
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load("en_core_web_sm")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_dynamic"
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# PREPROCESSING PIPELINES
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens]

def stem(tokens):
    return [stemmer.stem(w) for w in tokens]

def pos_lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def pipeline_process(text, stage):
    if stage == "raw":
        return text.lower()
    elif stage == "clean":
        return clean_text(text)
    elif stage == "token":
        return " ".join(tokenize(clean_text(text)))
    elif stage == "stop":
        tokens = tokenize(clean_text(text))
        tokens = remove_stopwords(tokens)
        return " ".join(tokens)
    elif stage == "lemma":
        tokens = tokenize(clean_text(text))
        tokens = lemmatize(tokens)
        return " ".join(tokens)
    elif stage == "stem":
        tokens = tokenize(clean_text(text))
        tokens = stem(tokens)
        return " ".join(tokens)
    elif stage == "poslemma":
        return " ".join(pos_lemmatize(clean_text(text)))

# LOAD AND SPLIT PDF DATA
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs.")
    return documents

def create_chunks(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(extracted_data)
    print(f"Created {len(chunks)} chunks from text.")
    return chunks

# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

# TEST QUERIES (for evaluation)
test_queries = [
    ("What are the side effects of metformin?", ["metformin", "diabetes"]),
    ("Treatment options for hypertension?", ["hypertension", "blood pressure"]),
    ("How does insulin regulate blood sugar?", ["insulin", "glucose"]),
    ("Symptoms of chronic kidney disease?", ["kidney", "disease"]),
    ("What does amoxicillin treat?", ["amoxicillin", "infection"]),
    ("How to manage asthma?", ["asthma", "airway", "breath"]),
]

# EVALUATION FUNCTION
def evaluate_pipeline(text_chunks, stage):
    processed_docs = [
        Document(page_content=pipeline_process(doc.page_content, stage))
        for doc in text_chunks[:50]  # limit for speed
    ]
    db = FAISS.from_documents(processed_docs, embedding_model)

    y_true, y_pred = [], []
    for q, keywords in test_queries:
        res = db.similarity_search(q, k=3)
        retrieved_texts = " ".join([r.page_content.lower() for r in res])
        tokens = set(retrieved_texts.split())

        for kw in keywords:
            y_true.append(1)
            y_pred.append(1 if kw in tokens else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

# MAIN WORKFLOW
def main():
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)

    pipelines = ["raw", "clean", "token", "stop", "lemma", "stem", "poslemma"]
    results = []

    print("\nEvaluating all preprocessing pipelines...\n")
    for p in tqdm(pipelines):
        pr, rc, f1, acc = evaluate_pipeline(text_chunks, p)
        results.append((p, pr, rc, f1, acc))

    # Print summary
    print("\nPipeline Evaluation Summary:")
    print(f"{'Pipeline':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Accuracy':<12}")
    for p, pr, rc, f1, acc in results:
        print(f"{p:<12}{pr:<12.3f}{rc:<12.3f}{f1:<12.3f}{acc:<12.3f}")

    # Pick best pipeline (based on F1 score)
    best_pipeline = max(results, key=lambda x: x[3])[0]
    print(f"\nBest pipeline: {best_pipeline.upper()} (based on F1 score)\n")

    # Rebuild final FAISS using best pipeline
    print("Building final FAISS vector store using best pipeline...")
    processed_docs = [
        Document(page_content=pipeline_process(doc.page_content, best_pipeline))
        for doc in text_chunks
    ]
    final_db = FAISS.from_documents(processed_docs, embedding_model)
    final_db.save_local(DB_FAISS_PATH)
    print(f"Saved optimized FAISS vector store at: {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()
