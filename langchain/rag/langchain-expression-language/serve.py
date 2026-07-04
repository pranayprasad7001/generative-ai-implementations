import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes

# Load environment variables from a .env file
load_dotenv()

# Initialize the Groq LLM client using the fast Llama 3.1 8B model
model = ChatGroq(model="llama-3.1-8b-instant")

# Define the system instructions for translation and structure the chat prompt layout
system_template = "Translate the following language into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)

# Instantiate a string parser to convert the raw LLM chat message output into a clean string
parser = StrOutputParser()

# Construct the execution pipeline: Prompt -> LLM -> String Output Parser
chain = prompt_template | model | parser

# Initialize FastAPI with metadata that will automatically populate the Swagger UI docs
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API Server using Langchain runnable interface",
)

# Expose the LCEL chain as fully-functional REST API endpoints (creates /chain/invoke, /chain/stream, etc.)
add_routes(app, chain, path="/chain")

# Start the Uvicorn ASGI server to host the FastAPI application locally on port 8000
if __name__ == "__main__":
    uvicorn.run("serve:app", host="localhost", port=8000, reload=True)