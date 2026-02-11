import os, re, string
import nltk, spacy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# INITIAL SETUP
load_dotenv()
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load("en_core_web_sm")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_sequential"
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# FULL NLP PREPROCESSING PIPELINE
def preprocess_full_pipeline(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())        # clean non-letters
    tokens = nltk.word_tokenize(text)                    # tokenize
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # lemmatize
    tokens = [stemmer.stem(w) for w in tokens]           # stem
    doc = nlp(" ".join(tokens))                          # POS-based lemmatization
    tokens = [token.lemma_ for token in doc]
    processed_text = " ".join(tokens)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()
    return processed_text

# LOAD & SPLIT PDF
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from your book(s).")
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks for embedding.")
    return chunks

# EMBEDDING MODEL (Medical)
embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

# EVALUATION QUERIES
test_queries = [
    ("What are the side effects of metformin?", ["metformin", "diabetes"]),
    ("Treatment options for hypertension?", ["hypertension", "blood", "pressure"]),
    ("How does insulin regulate blood sugar?", ["insulin", "glucose"]),
    ("Symptoms of chronic kidney disease?", ["kidney", "disease"]),
    ("What does amoxicillin treat?", ["amoxicillin", "infection"]),
    ("How to manage asthma?", ["asthma", "airway", "breath"]),
    ("What are antibiotics used for?", ["antibiotic", "infection"]),
    ("Causes of fever?", ["fever", "infection", "inflammation"]),
]

# MAIN PIPELINE + METRICS
def main():
    print("\n🔹 Starting Sequential Preprocessing + FAISS Builder...")
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)

    # Sequentially preprocess all chunks
    print("\n🔹 Running preprocessing on text chunks...")
    processed_docs = []
    for doc in tqdm(text_chunks, desc="Processing"):
        cleaned_text = preprocess_full_pipeline(doc.page_content)
        processed_docs.append(Document(page_content=cleaned_text))

    # Build FAISS DB
    print("\n🔹 Building FAISS vector database...")
    db = FAISS.from_documents(processed_docs, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Saved FAISS store at: {DB_FAISS_PATH}")

    # Run evaluation
    print("\n🔹 Evaluating retrieval performance...")
    y_true, y_pred = [], []

    for q, keywords in tqdm(test_queries, desc="Evaluating"):
        results = db.similarity_search(q, k=3)
        retrieved_texts = " ".join([r.page_content.lower() for r in results])
        tokens = set(retrieved_texts.split())

        for kw in keywords:
            y_true.append(1)
            y_pred.append(1 if kw in tokens else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nEvaluation Metrics on Sequential Pipeline:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # ==============================
    # VISUALIZATION
    # ==============================
    plt.figure(figsize=(6,4))
    plt.bar(["Precision", "Recall", "F1 Score"], [precision, recall, f1], color=["#007bff", "#28a745", "#ff9800"])
    plt.title("Sequential NLP Pipeline Performance on Medical Book")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.show()


if __name__ == "__main__":
    main()
