import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables (for API keys)
load_dotenv()

# Load the FAISS index
embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# Initialize LLM (Groq + Mixtral)
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"
)

retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit App UI
st.set_page_config(page_title="📄 PDF Chatbot", page_icon="🤖")
st.title("📄 Chat with your PDF")

question = st.text_input("Ask a question about your PDF")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(question)
        st.markdown(f"**Answer:** {response}")
