import re, os, string
import nltk, spacy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# INITIAL SETUP
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load("en_core_web_sm")

DATA_PATH = "data/"
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# PREPROCESSING PIPELINES
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
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

# LOAD BOOK & SPLIT
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from book(s).")
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return text_splitter.split_documents(documents)

documents = load_pdf_files(DATA_PATH)
text_chunks = create_chunks(documents)

# EVALUATION SETUP
test_queries = [
    ("What are the side effects of metformin?", ["metformin", "diabetes"]),
    ("Treatment options for hypertension?", ["hypertension", "blood pressure"]),
    ("How does insulin regulate blood sugar?", ["insulin", "glucose"]),
    ("Symptoms of chronic kidney disease?", ["kidney", "disease"]),
    ("What does amoxicillin treat?", ["amoxicillin", "infection"]),
    ("How to manage asthma?", ["asthma", "airway", "breath"]),
    ("What are antibiotics used for?", ["antibiotic", "infection"]),
    ("Causes of fever?", ["fever", "infection", "inflammation"]),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

pipelines = ["raw", "clean", "token", "stop", "lemma", "stem", "poslemma"]
results = []

# EVALUATION FUNCTION
def evaluate_pipeline(stage):
    processed_docs = []
    for doc in text_chunks[:60]:  # limit for speed, can increase
        processed_text = pipeline_process(doc.page_content, stage)
        processed_docs.append(Document(page_content=processed_text))

    db = FAISS.from_documents(processed_docs, embedding_model)

    y_true, y_pred = [], []
    for q, keywords in test_queries:
        res = db.similarity_search(q, k=3)
        retrieved_texts = " ".join([r.page_content.lower() for r in res])
        tokens = set(retrieved_texts.split())

        for kw in keywords:
            y_true.append(1)
            y_pred.append(1 if kw in tokens else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# RUN EXPERIMENTS
print("\nEvaluating pipelines on book data...\n")
for p in tqdm(pipelines, desc="Testing pipelines"):
    p_, r_, f_ = evaluate_pipeline(p)
    results.append((p, p_, r_, f_))

# DISPLAY & VISUALIZE
print("\nNLP Pipeline Evaluation Results on Your Book:")
print("-------------------------------------------------------------")
print(f"{'Pipeline':<15}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")
for p, pr, re, f in results:
    print(f"{p:<15}{pr:<12.3f}{re:<12.3f}{f:<12.3f}")

# Bar chart
pipenames = [r[0] for r in results]
precisions = [r[1] for r in results]
recalls = [r[2] for r in results]
f1s = [r[3] for r in results]

x = np.arange(len(pipelines))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precisions, width, label="Precision")
plt.bar(x, recalls, width, label="Recall")
plt.bar(x + width, f1s, width, label="F1 Score")
plt.xticks(x, pipenames)
plt.ylabel("Score")
plt.title("NLP Preprocessing Pipeline Performance on Medical Book")
plt.legend()
plt.ylim(0, 1)
plt.show()
