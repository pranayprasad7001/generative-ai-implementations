"""
Multimodal RAG with LangChain - Complete Example
Processes PDFs containing text, tables, and images.

Architecture Overview

Extract → unstructured partitions PDF into text, tables, and images
Summarize → GPT-4o generates text summaries of images/tables
Index → MultiVectorRetriever stores summaries in ChromaDB, raw content in docstore
Retrieve → Semantic search on summaries returns original images/text
Generate → GPT-4o receives both text context and base64 images for the final answer

Required Dependencies

pip install -U langchain langchain-openai langchain-chroma langchain-text-splitters
pip install "unstructured[all-docs]" pdf2image pytesseract
pip install pillow chromadb tiktoken

"""

import os
import uuid
import base64
import io
from typing import List, Dict, Any

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# LangChain integrations
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document parsing
from unstructured.partition.pdf import partition_pdf
from PIL import Image

# Setup API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Paths
PDF_PATH = "data/sample.pdf"
IMAGE_OUTPUT_DIR = "data/images"


# =============================================================================
# STEP 1: EXTRACT PDF ELEMENTS (Text, Tables, Images)
# =============================================================================

def extract_pdf_elements(pdf_path: str, image_output_dir: str):
    """
    Use unstructured to extract text, tables, and images from PDF.
    Images are saved to disk; text/tables are returned as objects.
    """
    os.makedirs(image_output_dir, exist_ok=True)
    
    raw_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,           # Extract embedded images
        infer_table_structure=True,           # Detect tables
        chunking_strategy="by_title",         # Chunk by document sections
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_output_dir,
    )
    return raw_elements


def categorize_elements(raw_elements):
    """
    Separate extracted elements into texts, tables, and image paths.
    """
    texts = []
    tables = []
    image_paths = []
    
    for element in raw_elements:
        elem_type = str(type(element))
        
        if "unstructured.documents.elements.Table" in elem_type:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in elem_type:
            texts.append(str(element))
        elif "unstructured.documents.elements.Image" in elem_type:
            # Image elements contain path in metadata
            if hasattr(element, 'metadata') and element.metadata.image_path:
                image_paths.append(element.metadata.image_path)
    
    return texts, tables, image_paths


# =============================================================================
# STEP 2: SUMMARIZE CONTENT WITH GPT-4o
# =============================================================================

def summarize_text(texts: List[str]) -> List[str]:
    """Generate concise summaries of text chunks."""
    prompt = ChatPromptTemplate.from_template("""
    Summarize the following text chunk concisely for retrieval purposes.
    Capture key facts, entities, and concepts:
    
    {text}
    """)
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | model | StrOutputParser()
    
    return chain.batch(texts, {"max_concurrency": 5})


def summarize_tables(tables: List[str]) -> List[str]:
    """Generate summaries of table content."""
    prompt = ChatPromptTemplate.from_template("""
    Summarize the following table. Describe what data it contains,
    key columns, and any notable values or trends:
    
    {table}
    """)
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | model | StrOutputParser()
    
    return chain.batch(tables, {"max_concurrency": 5})


def summarize_images(image_paths: List[str]) -> List[str]:
    """
    Use GPT-4o Vision to generate text descriptions of images.
    These descriptions will be embedded for retrieval.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    summaries = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                Describe this image in detail. Include:
                - What type of visual it is (chart, diagram, photo, screenshot)
                - All visible text, labels, and numbers
                - Key objects, relationships, and trends
                - If it's a chart, describe axes, data points, and conclusions
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        
        response = model.invoke([message])
        summaries.append(response.content)
    
    return summaries


# =============================================================================
# STEP 3: BUILD MULTI-VECTOR RETRIEVER
# =============================================================================

def create_multimodal_retriever(
    text_summaries, texts,
    table_summaries, tables,
    image_summaries, image_paths
):
    """
    Create a MultiVectorRetriever that indexes summaries
    but returns the original raw content.
    """
    # Vector store for summaries (semantic search)
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="data/chroma_db"
    )
    
    # Docstore for original content
    docstore = InMemoryStore()
    
    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": 6}  # Retrieve top 6 matches
    )
    
    id_key = "doc_id"
    
    def add_documents(retriever, summaries, contents):
        """Helper: add summaries to vectorstore, originals to docstore."""
        doc_ids = [str(uuid.uuid4()) for _ in contents]
        
        # Summary documents go to vectorstore (for search)
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        
        # Original content goes to docstore (for retrieval)
        retriever.docstore.mset(list(zip(doc_ids, contents)))
    
    # Add all modalities
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    
    if image_summaries:
        # For images, store the image path as the retrievable content
        add_documents(retriever, image_summaries, image_paths)
    
    return retriever


# =============================================================================
# STEP 4: MULTIMODAL RAG CHAIN
# =============================================================================

def prepare_context(docs):
    """
    Split retrieved documents into text and image paths.
    Load images as base64 for the multimodal LLM.
    """
    text_contexts = []
    image_base64_list = []
    
    for doc in docs:
        # If it's a path that exists and is an image
        if isinstance(doc, str) and os.path.exists(doc):
            # Check if image
            try:
                with open(doc, "rb") as f:
                    img_data = f.read()
                # Verify it's an image
                Image.open(io.BytesIO(img_data))
                image_base64_list.append(base64.b64encode(img_data).decode("utf-8"))
            except Exception:
                text_contexts.append(doc)
        else:
            text_contexts.append(str(doc))
    
    return {
        "texts": text_contexts,
        "images": image_base64_list
    }


def build_multimodal_rag_chain(retriever):
    """
    Build the end-to-end RAG chain.
    Retrieves context → Prepares multimodal prompt → Generates answer.
    """
    # Multimodal LLM for final generation
    model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=2048)
    
    # Retrieval step
    retrieve_context = (
        retriever 
        | RunnableLambda(prepare_context)
    )
    
    # Prompt builder
    def build_prompt(inputs):
        question = inputs["question"]
        context = inputs["context"]
        
        # System instruction
        system_msg = SystemMessage(content="""
        You are an expert research assistant analyzing documents.
        Answer the user's question using the provided text and image context.
        When images are provided, analyze them carefully and reference specific details.
        If information is insufficient, say so clearly.
        """)
        
        # Build content list
        content = [
            {"type": "text", "text": f"Question: {question}\n\nText Context:\n{chr(10).join(context['texts'])}"}
        ]
        
        # Add images to the message
        for img_b64 in context["images"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        human_msg = HumanMessage(content=content)
        return [system_msg, human_msg]
    
    # Full chain
    chain = (
        RunnableParallel(
            {
                "context": retrieve_context,
                "question": RunnablePassthrough()
            }
        )
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
    
    return chain


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("Multimodal RAG Pipeline")
    print("=" * 60)
    
    # 1. Extract
    print("\n[1/5] Extracting PDF elements...")
    raw_elements = extract_pdf_elements(PDF_PATH, IMAGE_OUTPUT_DIR)
    texts, tables, image_paths = categorize_elements(raw_elements)
    print(f"   Found: {len(texts)} text chunks, {len(tables)} tables, {len(image_paths)} images")
    
    # 2. Summarize
    print("\n[2/5] Summarizing content with GPT-4o...")
    text_summaries = summarize_text(texts) if texts else []
    table_summaries = summarize_tables(tables) if tables else []
    image_summaries = summarize_images(image_paths) if image_paths else []
    print("   Summaries generated")
    
    # 3. Build Retriever
    print("\n[3/5] Building MultiVectorRetriever...")
    retriever = create_multimodal_retriever(
        text_summaries, texts,
        table_summaries, tables,
        image_summaries, image_paths
    )
    print("   Index built and persisted")
    
    # 4. Build Chain
    print("\n[4/5] Building RAG chain...")
    rag_chain = build_multimodal_rag_chain(retriever)
    
    # 5. Query
    print("\n[5/5] Ready for queries!")
    print("-" * 60)
    
    # Example queries
    queries = [
        "What are the key findings in the document?",
        "Explain the chart showing revenue trends",
        "Compare the values in the tables",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        answer = rag_chain.invoke(query)
        print(f"💡 Answer: {answer[:500]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()