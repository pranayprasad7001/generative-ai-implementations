"""
RAG retrieval pipeline reference implementation (packaged version)
====================================================================

Implements the pipeline:

    Hybrid retrieval (RRF *or* weighted linear fusion)
            -> Reranking (CrossEncoderReranker)
            -> MMR (FAISS's built-in max_marginal_relevance_search)
            -> LLM generation

RRF and weighted linear fusion are ALTERNATIVES for stage 1, not both used
at once -- pick whichever fits your infra:
    - RRF:      two separate retrievers (FAISS + BM25), merged via
                LangChain's EnsembleRetriever. Fully local, no extra service.
    - weighted: a single native hybrid query against Pinecone, which is
                what "weighted linear vector fusion" actually refers to in
                practice (Qdrant/Weaviate work the same way).

Every stage below uses a package/library component rather than a hand-rolled
reimplementation:
    - RRF        -> langchain_classic EnsembleRetriever
    - weighted   -> Pinecone native hybrid search (dotproduct index)
    - reranking  -> langchain_classic CrossEncoderReranker + HuggingFaceCrossEncoder
    - MMR        -> FAISS.max_marginal_relevance_search

Requirements:
    pip install langchain langchain-classic langchain-community \
                langchain-huggingface sentence-transformers rank_bm25 \
                faiss-cpu pinecone pinecone-text numpy

Env vars:
    GROQ_API_KEY      (or swap init_chat_model for your provider of choice)
    PINECONE_API_KEY  (only needed for the weighted-fusion path)
"""

import os
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.prompts import PromptTemplate

from langchain.chat_models import init_chat_model

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ---------------------------------------------------------------------------
# Sample corpus -- swap this for your own loader (TextLoader, PyPDFLoader, etc.)
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(page_content="LangChain helps build LLM applications by providing "
                           "abstractions for prompts, chains, memory, and agents."),
    Document(page_content="FAISS is a library for fast approximate nearest neighbor "
                           "search, commonly used for dense vector retrieval."),
    Document(page_content="BM25 is a sparse, keyword-based retrieval algorithm that "
                           "scores documents by term frequency and rarity."),
    Document(page_content="Hybrid search combines dense vector search with sparse "
                           "keyword search to get the strengths of both."),
    Document(page_content="Reciprocal Rank Fusion merges ranked lists from multiple "
                           "retrievers using only their rank positions, not raw scores."),
    Document(page_content="Weighted linear fusion scales dense and sparse vectors by "
                           "an alpha factor before a single combined similarity score."),
    Document(page_content="Cross-encoder rerankers jointly encode the query and each "
                           "candidate document to produce a more accurate relevance score."),
    Document(page_content="Maximal Marginal Relevance removes redundant chunks from "
                           "the final context by balancing relevance against diversity."),
    Document(page_content="RAG pipelines retrieve relevant context and pass it to an "
                           "LLM so answers are grounded in real documents."),
    Document(page_content="Pinecone, Qdrant, and Weaviate support native single-pass "
                           "hybrid queries by combining dense and sparse vectors directly."),
]

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Stage 1a: Reciprocal Rank Fusion  (local, FAISS + BM25)
#   RRF(d) = sum over retrievers r of  weight_r / (k + rank_r(d))
# ---------------------------------------------------------------------------

@dataclass
class LocalRetrievers:
    dense_retriever: object
    bm25_retriever: BM25Retriever


def build_local_retrievers(docs: list[Document]) -> LocalRetrievers:
    vectorstore = FAISS.from_documents(docs, embedding_model)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": len(docs)})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = len(docs)

    return LocalRetrievers(dense_retriever=dense_retriever, bm25_retriever=bm25_retriever)


def hybrid_retrieve_rrf(query: str, retrievers: LocalRetrievers, top_k: int = 4,
                         weights: tuple[float, float] = (0.7, 0.3)) -> list[Document]:
    ensemble = EnsembleRetriever(
        retrievers=[retrievers.dense_retriever, retrievers.bm25_retriever],
        weights=list(weights),  # (dense_weight, sparse_weight)
    )
    return ensemble.invoke(query)[:top_k]


# ---------------------------------------------------------------------------
# Stage 1b: Weighted Linear Vector Fusion  (Pinecone native hybrid search)
#   score(q, d) = alpha * sim_dense(q, d) + (1 - alpha) * sim_sparse(q, d)
#   Pinecone combines a scaled dense + sparse vector in a single dotproduct
#   query -- this is the "native single-pass hybrid" the technique refers to.
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME = "hybrid-search-rag-pipeline"


def setup_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",  # required for sparse_values support
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def build_sparse_encoder(docs: list[Document]) -> BM25Encoder:
    # .fit() on your own corpus (rather than .default()'s pretrained MS MARCO
    # stats) so the IDF weights actually match your documents.
    encoder = BM25Encoder()
    encoder.fit([d.page_content for d in docs])
    return encoder


def upsert_documents(index_obj, docs: list[Document], sparse_encoder: BM25Encoder) -> None:
    texts = [d.page_content for d in docs]
    dense_vectors = embedding_model.embed_documents(texts)
    sparse_vectors = sparse_encoder.encode_documents(texts)

    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": dense,
            "sparse_values": sparse,
            "metadata": {"text": text},
        }
        for text, dense, sparse in zip(texts, dense_vectors, sparse_vectors)
    ]
    index_obj.upsert(vectors=vectors)


def hybrid_scale(dense: list[float], sparse: dict, alpha: float):
    """Convex combination: alpha * dense + (1 - alpha) * sparse.
    alpha = 1.0 -> pure dense (semantic). alpha = 0.0 -> pure sparse (keyword)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def hybrid_retrieve_weighted(query: str, index_obj, sparse_encoder: BM25Encoder,
                              top_k: int = 4, alpha: float = 0.5) -> list[Document]:
    dense_query = embedding_model.embed_query(query)
    sparse_query = sparse_encoder.encode_queries(query)
    scaled_dense, scaled_sparse = hybrid_scale(dense_query, sparse_query, alpha)

    results = index_obj.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    return [Document(page_content=match["metadata"]["text"]) for match in results["matches"]]


# ---------------------------------------------------------------------------
# Stage 2: Reranking  (CrossEncoderReranker package)
#   A cross-encoder jointly scores (query, document) pairs -- more accurate
#   than comparing independent embeddings, at the cost of one extra
#   inference per candidate.
# ---------------------------------------------------------------------------

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
# Swap to "BAAI/bge-reranker-v2-m3" for multilingual / higher-accuracy reranking.


def rerank_documents(query: str, docs: list[Document], top_n: int = 4) -> list[Document]:
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return list(reranker.compress_documents(documents=docs, query=query))


# ---------------------------------------------------------------------------
# Stage 3: Maximal Marginal Relevance  (FAISS built-in)
#   MMR = argmax_i [ lambda * Sim(d_i, q) - (1 - lambda) * max_j Sim(d_i, d_j) ]
#   Runs on the RERANKED list -- build a small ephemeral FAISS index over
#   just those candidates and let FAISS's own MMR search do the selection.
# ---------------------------------------------------------------------------

def mmr_filter(query: str, docs: list[Document], top_k: int = 3,
                lambda_mult: float = 0.5) -> list[Document]:
    if len(docs) <= top_k:
        return docs
    temp_store = FAISS.from_documents(docs, embedding_model)
    return temp_store.max_marginal_relevance_search(
        query, k=top_k, fetch_k=len(docs), lambda_mult=lambda_mult
    )


# ---------------------------------------------------------------------------
# Stage 4: LLM generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = PromptTemplate.from_template("""
Answer the question based only on the context provided.

Context:
{context}

Question: {question}
""")


def generate_answer(query: str, docs: list[Document], llm) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    chain = GENERATION_PROMPT | llm
    response = chain.invoke({"context": context, "question": query})
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(query: str, llm, fusion_method: str = "rrf",
                  local_retrievers: LocalRetrievers | None = None,
                  pinecone_index=None, sparse_encoder: BM25Encoder | None = None,
                  fetch_k: int = 6, rerank_k: int = 4, final_k: int = 3,
                  alpha: float = 0.5, lambda_mult: float = 0.5) -> dict:
    if fusion_method == "rrf":
        candidates = hybrid_retrieve_rrf(query, local_retrievers, top_k=fetch_k)
    elif fusion_method == "weighted":
        candidates = hybrid_retrieve_weighted(query, pinecone_index, sparse_encoder,
                                               top_k=fetch_k, alpha=alpha)
    else:
        raise ValueError(f"Unknown method '{fusion_method}', expected 'rrf' or 'weighted'")

    reranked = rerank_documents(query, candidates, top_n=rerank_k)
    diverse = mmr_filter(query, reranked, top_k=final_k, lambda_mult=lambda_mult)
    answer = generate_answer(query, diverse, llm)

    return {
        "query": query,
        "candidates": candidates,
        "reranked": reranked,
        "final_context": diverse,
        "answer": answer,
    }


def print_result(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print("Final context used:")
    for i, d in enumerate(result["final_context"], 1):
        print(f"{i}. {d.page_content}")
    print("\nAnswer:\n", result["answer"])


if __name__ == "__main__":
    llm = init_chat_model("groq:gemma2-9b-it")  # swap for your provider/model
    query = "How do hybrid search and reranking improve RAG?"

    # --- RRF path: fully local, no external service needed ---
    local_retrievers = build_local_retrievers(SAMPLE_DOCS)
    rrf_result = run_pipeline(query, llm, fusion_method="rrf", local_retrievers=local_retrievers)
    print_result("RRF pipeline (FAISS + BM25 via EnsembleRetriever)", rrf_result)

    # --- Weighted fusion path: needs a Pinecone account ---
    if PINECONE_API_KEY:
        sparse_encoder = build_sparse_encoder(SAMPLE_DOCS)
        pinecone_index = setup_pinecone_index()
        upsert_documents(pinecone_index, SAMPLE_DOCS, sparse_encoder)

        weighted_result = run_pipeline(query, llm, fusion_method="weighted",
                                        pinecone_index=pinecone_index,
                                        sparse_encoder=sparse_encoder, alpha=0.6)
        print_result("Weighted linear fusion pipeline (Pinecone native hybrid)", weighted_result)
    else:
        print("\n(Skipping weighted-fusion demo -- set PINECONE_API_KEY to run it)")
