import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


load_dotenv()
nvidia_nim_api_key = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") 
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) 
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) 
        print("Hello")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



st.title("NVIDIA NIM DEMO")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=nvidia_nim_api_key)

if st.button("Embed Documents"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

query = st.text_input("Enter Your Question From Doduments")

if query:

    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':query})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
