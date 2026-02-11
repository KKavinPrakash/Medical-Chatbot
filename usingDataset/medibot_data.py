# medibot_dataset.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
DB_PATH = "../vectorstore/medquad_faiss_sequential" 

@st.cache_resource
def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    db = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)
    return db

def main():
    st.title("MedQuad Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_prompt = st.chat_input("Ask medical question (dataset-backed)")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role":"user","content":user_prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore")
                return

            # LLM setup (Groq)
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"
            llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.2, max_tokens=512, api_key=GROQ_API_KEY)

            # Use retrieval qa chat prompt or custom
            retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_chain = create_stuff_documents_chain(llm, retrieval_prompt)

            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k":4}), combine_chain)
            response = rag_chain.invoke({"input": user_prompt})
            answer = response["answer"]

            # Show sources (if any) from the context returned
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})

            # display source metadata if present
            if "context" in response:
                st.write("**Sources:**")
                for d in response["context"]:
                    meta = d.metadata if hasattr(d, "metadata") else {}
                    src = meta.get("source", "unknown")
                    focus = meta.get("focus_area","")
                    st.write(f"- {src} | {focus}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
