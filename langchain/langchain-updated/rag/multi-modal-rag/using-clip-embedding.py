"""
LangChain Unified Multimodal RAG
Equivalent to LlamaIndex's MultiModalVectorStoreIndex
Uses CLIP embeddings for both text and images in a single vector space.
"""

import os
import base64
import io
from pathlib import Path
from typing import List, Union

from PIL import Image

# LangChain
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# Multimodal embeddings — the key piece
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# Vector store
from langchain_chroma import Chroma

# LLM
from langchain_openai import ChatOpenAI

# PDF parsing
from unstructured.partition.pdf import partition_pdf


os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

PDF_PATH = "data/sample.pdf"
IMAGE_OUTPUT_DIR = "data/extracted_images"


# =============================================================================
# STEP 1: EXTRACT ELEMENTS FROM PDF
# =============================================================================

def extract_pdf_elements(pdf_path: str, image_output_dir: str):
    """Extract text chunks and images from PDF using unstructured."""
    os.makedirs(image_output_dir, exist_ok=True)
    
    elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_output_dir,
    )
    return elements


def categorize_elements(elements):
    """Separate into texts and image file paths."""
    texts = []
    image_paths = []
    
    for elem in elements:
        elem_type = str(type(elem))
        if "unstructured.documents.elements.Table" in elem_type:
            texts.append(f"[TABLE]\n{str(elem)}")
        elif "unstructured.documents.elements.CompositeElement" in elem_type:
            texts.append(str(elem))
        elif "unstructured.documents.elements.Image" in elem_type:
            if hasattr(elem, 'metadata') and elem.metadata.image_path:
                image_paths.append(elem.metadata.image_path)
    
    return texts, image_paths


# =============================================================================
# STEP 2: BUILD UNIFIED MULTIMODAL INDEX (The LlamaIndex Equivalent)
# =============================================================================

def build_multimodal_index(texts: List[str], image_paths: List[str]):
    """
    Create a unified vector store where BOTH text and images are embedded
    using the same CLIP model into the SAME vector space.
    
    This is the LangChain equivalent of LlamaIndex's MultiModalVectorStoreIndex.
    """
    
    # Initialize CLIP embeddings — handles both text and images
    # Default: ViT-H-14, laion2b_s32b_b79k (good balance of speed/quality)
    embeddings = OpenCLIPEmbeddings(
        model_name="ViT-H-14",
        checkpoint="laion2b_s32b_b79k"
    )
    
    # Single vector store for ALL modalities
    vectorstore = Chroma(
        collection_name="unified_multimodal_index",
        embedding_function=embeddings,
        persist_directory="data/chroma_multimodal"
    )
    
    # --- Index Text ---
    text_docs = [
        Document(
            page_content=text,
            metadata={"doc_type": "text", "source_index": i}
        )
        for i, text in enumerate(texts)
    ]
    vectorstore.add_documents(text_docs)
    print(f"Indexed {len(text_docs)} text chunks")
    
    # --- Index Images ---
    # Chroma's add_images() embeds images directly via CLIP
    if image_paths:
        image_metadatas = [
            {"doc_type": "image", "source_path": path, "source_index": i}
            for i, path in enumerate(image_paths)
        ]
        vectorstore.add_images(
            uris=image_paths,
            metadatas=image_metadatas
        )
        print(f"Indexed {len(image_paths)} images")
    
    return vectorstore


# =============================================================================
# STEP 3: MULTIMODAL RETRIEVAL & GENERATION CHAIN
# =============================================================================

def prepare_multimodal_context(docs: List[Document]):
    """
    Separate retrieved documents into text content and base64 images.
    Images stored in Chroma via CLIP have their content as base64 strings.
    """
    texts = []
    image_base64s = []
    
    for doc in docs:
        doc_type = doc.metadata.get("doc_type", "text")
        
        if doc_type == "image":
            # For image docs from Chroma+CLIP, page_content is often the image path
            # or base64 depending on version. We load from source_path.
            img_path = doc.metadata.get("source_path")
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    image_base64s.append(base64.b64encode(f.read()).decode("utf-8"))
        else:
            texts.append(doc.page_content)
    
    return {"texts": texts, "images": image_base64s}


def build_rag_chain(vectorstore):
    """Build the end-to-end multimodal RAG chain."""
    
    # Retriever from unified store — searches across text AND images
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # Returns mix of text docs and images
    )
    
    # Multimodal LLM for answer synthesis
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=2048)
    
    def build_prompt(inputs):
        question = inputs["question"]
        context = inputs["context"]
        
        system_msg = SystemMessage(content="""
        You are an expert research assistant. Answer using the provided text
        and image context. When images are included, analyze them carefully
        and reference specific visual details, charts, or diagrams.
        """)
        
        content = [{"type": "text", "text": f"Question: {question}\n\nText Context:\n{chr(10).join(context['texts'])}"}]
        
        for img_b64 in context["images"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        return [system_msg, HumanMessage(content=content)]
    
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(prepare_multimodal_context),
            "question": RunnablePassthrough()
        })
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
    
    return chain


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LangChain Unified Multimodal RAG")
    print("(Equivalent to LlamaIndex MultiModalVectorStoreIndex)")
    print("=" * 60)
    
    # 1. Extract
    print("\n[1/4] Extracting PDF elements...")
    elements = extract_pdf_elements(PDF_PATH, IMAGE_OUTPUT_DIR)
    texts, image_paths = categorize_elements(elements)
    print(f"   Texts: {len(texts)}, Images: {len(image_paths)}")
    
    # 2. Build unified index
    print("\n[2/4] Building unified multimodal index with CLIP...")
    vectorstore = build_multimodal_index(texts, image_paths)
    
    # 3. Build chain
    print("\n[3/4] Building RAG chain...")
    rag_chain = build_rag_chain(vectorstore)
    
    # 4. Query
    print("\n[4/4] Running queries...")
    queries = [
        "What are the key trends shown in the charts?",
        "Explain the relationship between the data in the tables and the diagrams",
        "Summarize the main findings from the document",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        answer = rag_chain.invoke(query)
        print(f"💡 Answer: {answer[:600]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()