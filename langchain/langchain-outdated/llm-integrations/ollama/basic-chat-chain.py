# Import Libraries
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables
load_dotenv()

# Cache the llm to prevent repeated model wrapper creation
@st.cache_resource
def load_model():
    return ChatOllama(model="qwen3:8b")

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful AI assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit Framework
st.title("Langchain Demo With Qwen3")
input_text = st.text_input("What question you have in your mind? ")

# Loading the ollama qwen3 model
llm_model = load_model()

# Initializing the output parser
output_parser = StrOutputParser()

# Creating the chain
chain = prompt | llm_model | output_parser

# Invoking the chain with user's input for response
if input_text:
    st.write(chain.invoke({"question": input_text}))