import os
import pickle
import streamlit as st

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

VECTORLESS_INDEX_PATH = "vectorstore/bm25_index.pkl"

@st.cache_resource
def get_vectorless_retriever():
    if not os.path.exists(VECTORLESS_INDEX_PATH):
        return None
    with open(VECTORLESS_INDEX_PATH, 'rb') as f:
        retriever = pickle.load(f)
    return retriever

def main():
    st.title("Ask Chatbot! (Vectorless DB Native)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        
        # Build chat history for LangChain
        chat_history = []
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            else:
                chat_history.append(AIMessage(content=msg['content']))

        st.session_state.messages.append({'role':'user', 'content': prompt})
                
        try: 
            retriever = get_vectorless_retriever()
            if retriever is None:
                st.error("Failed to load the Vectorless DB. Please run create_memory_vectorless.py first.")
                return

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"  
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            # 1. Contextualize question (History-aware retriever)
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            
            # 2. Answer question
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            
            # 3. Final RAG Chain
            rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

            response = rag_chain.invoke({
                'input': prompt,
                'chat_history': chat_history
            })

            result = response["answer"]
            
            # Show the pages we retrieved for context
            context_docs = response.get("context", [])
            page_numbers = set([str(doc.metadata.get("page", "?")) for doc in context_docs])
            
            if page_numbers:
                result += f"\n\n**Sources:** Document Pages {', '.join(page_numbers)}"

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
