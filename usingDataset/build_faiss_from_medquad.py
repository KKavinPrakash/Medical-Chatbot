# build_faiss_from_medquad.py
import os, re
import nltk, spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from ingest_medquad import load_medquad, df_to_documents

# Setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

DB_OUTPUT = "../vectorstore/medquad_faiss_sequential"
os.makedirs("vectorstore", exist_ok=True)

# Choose embedding model (change as desired)
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

def preprocess_full_pipeline(text: str) -> str:
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]
    processed = " ".join(tokens)
    return re.sub(r"\s+"," ", processed).strip()

def documents_from_df(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}\nFocus: {row.get('focus_area','')}"
        content_processed = preprocess_full_pipeline(content)
        metadata = {"source": row.get("source",""), "focus_area": row.get("focus_area","")}
        docs.append(Document(page_content=content_processed, metadata=metadata))
    return docs

def build_and_save_faiss(df):
    docs = documents_from_df(df)
    embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Building FAISS with {len(docs)} docs using {EMBEDDING_MODEL} ...")
    db = FAISS.from_documents(docs, embed)
    db.save_local(DB_OUTPUT)
    print("Saved FAISS at:", DB_OUTPUT)
    return db

if __name__ == "__main__":
    df = load_medquad()
    # optionally take only train split or whole dataset
    from ingest_medquad import create_train_eval_splits
    train, test = create_train_eval_splits(df, test_size=0.1)
    db = build_and_save_faiss(train)  # build from train set
