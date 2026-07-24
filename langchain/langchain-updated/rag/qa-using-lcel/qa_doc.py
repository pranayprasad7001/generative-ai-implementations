import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import TextLoader
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Configure your keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 1. Load and Split Documents
loader = TextLoader("your_knowledge_base.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 2. Persist Vector Store using Chroma
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the primary LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Configure the trimmer to preserve the last 2000 tokens
# Strategy "last" keeps recent exchanges; start_on="human" maintains structure integrity
trimmer = trim_messages(
    max_tokens=2000,
    strategy="last",
    token_counter=llm, # Dynamically tracks token counts for the specific model
    start_on="human",
    include_system=True # Keeps your system prompt anchored at the top
)

# Step A: Contextualize the prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Sub-chain to resolve ambiguous follow-up questions
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# Step B: Main Question-Answering Prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Helper function to format retrieved chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step C: The Composite RAG Chain Pipeline
def contextualized_retrieval(input_dict):
    # Pass chat history through the trimmer first to prevent context overflow
    trimmed_history = trimmer.invoke(input_dict["chat_history"])
    
    if trimmed_history:
        # If history exists, pass trimmed history to rewrite query
        return contextualize_q_chain.invoke({
            "chat_history": trimmed_history,
            "input": input_dict["input"]
        })
    return input_dict["input"]

# Core Engine Chain
rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_retrieval | retriever | format_docs
    )
    | qa_prompt
    | llm
)

# Session dictionary to temporarily persist in-memory across calls
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Wrap our core RAG chain with session capability
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Example usage

config = {"configurable": {"session_id": "user_session_abc123"}}

# Session Turn 1: Direct question ingestion 
response1 = conversational_rag_chain.invoke(
    {"input": "What are the company's rules on remote work?"},
    config=config
)
print("Bot:", response1.content)

# Session Turn 2: Follow-up relying entirely on memory and context rewriting
response2 = conversational_rag_chain.invoke(
    {"input": "Does this change during the summer months?"}, 
    config=config
)
print("Bot:", response2.content)
