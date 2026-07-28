import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import trim_messages
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

# Initialize the primary LLM (streaming is enabled by default in ChatOpenAI)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Configure the message trimmer
trimmer = trim_messages(
    max_tokens=2000,
    strategy="last",
    token_counter=llm,
    start_on="human",
    include_system=True
)

# Step A: Contextualize Prompt
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

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# Step B: Main QA Prompt
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step C: Contextual Retrieval Logic
def contextualized_retrieval(input_dict):
    history = input_dict.get("chat_history", [])
    if history:
        trimmed_history = trimmer.invoke(history)
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

# Session store for memory persistence
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Wrap core RAG chain with session capability
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Function to run queries with streaming output
def ask_question_stream(query: str, session_id: str):
    config = {"configurable": {"session_id": session_id}}
    print(f"User: {query}")
    print("Bot: ", end="", flush=True)

    # Use .stream() instead of .invoke()
    for chunk in conversational_rag_chain.stream({"input": query}, config=config):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n" + "-" * 50)

# Execution
if __name__ == "__main__":
    session_id = "user_session_abc123"

    # Turn 1
    ask_question_stream("What are the company's rules on remote work?", session_id)

    # Turn 2
    ask_question_stream("Does this change during the summer months?", session_id)