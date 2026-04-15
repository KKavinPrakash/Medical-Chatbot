import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
VECTORLESS_INDEX_PATH = "vectorstore/bm25_index.pkl"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    # PyPDFLoader already sets 'page' in metadata, but let's ensure it's there
    for i, doc in enumerate(documents):
        if 'page' not in doc.metadata:
            doc.metadata['page'] = i
    return documents

def main():
    print("Loading documents (each page is considered a document node)...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} pages")

    print("Creating BM25 Vectorless Page Index...")
    # By passing documents directly, we treat each page as an indexable node
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    print("Saving BM25 Vectorless DB to disk...")
    os.makedirs(os.path.dirname(VECTORLESS_INDEX_PATH), exist_ok=True)
    with open(VECTORLESS_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    
    print(f"Saved new Vectorless Index at {VECTORLESS_INDEX_PATH}")

if __name__ == "__main__":
    main()
