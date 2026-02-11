# create_memory_for_llm_medical.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_medical"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(extracted_data)

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    return embedding_model

def main():
    print("Loading documents...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} pages")

    print("Creating chunks...")
    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} text chunks")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector store...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Saved new medical vector store at {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()
