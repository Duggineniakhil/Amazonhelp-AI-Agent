"""
Embedding-based retriever using sentence-transformers + FAISS.

Embeds all customer messages from the processed dataset and builds a
FAISS index for fast similarity search. At query time, retrieves the
top-K most similar historical conversations to provide context for
reply drafting.
"""

import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    INDEX_DIR,
    TOP_K_RETRIEVAL,
    PROCESSED_DATA_DIR,
)


class Retriever:
    """
    Semantic retriever for customer support conversations.
    
    Uses sentence-transformers to embed customer messages and FAISS
    for fast nearest-neighbor search. Returns the most similar
    historical (customer_message, agent_reply) pairs.
    """
    
    def __init__(self, load_existing: bool = True):
        """
        Initialize the retriever.
        
        Args:
            load_existing: If True, load a pre-built index from disk.
                          If False, you must call build_index() first.
        """
        print("[→] Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []  # List of (customer_text, agent_text, thread_id) tuples
        
        if load_existing and FAISS_INDEX_PATH.exists():
            self.load_index()
    
    def build_index(self, pairs: list):
        """
        Build a FAISS index from (customer, agent) pairs.
        
        Args:
            pairs: List of dicts with 'customer_text' and 'agent_text' keys.
        """
        print(f"\n[→] Building FAISS index from {len(pairs):,} pairs...")
        
        # Extract customer texts for embedding
        customer_texts = [p["customer_text"] for p in pairs]
        
        # Embed in batches
        print("[→] Computing embeddings...")
        embeddings = self.model.encode(
            customer_texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )
        
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"    Embeddings shape: {embeddings.shape}")
        
        # Build FAISS index (inner product = cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        print(f"    Index size: {self.index.ntotal:,} vectors")
        
        # Store metadata
        self.metadata = [
            {
                "customer_text": p["customer_text"],
                "agent_text": p["agent_text"],
                "thread_id": p.get("thread_id", ""),
            }
            for p in pairs
        ]
        
        # Save to disk
        self.save_index(embeddings)
    
    def save_index(self, embeddings: np.ndarray = None):
        """Save the FAISS index and metadata to disk."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        print(f"[✓] FAISS index saved to {FAISS_INDEX_PATH}")
        
        if embeddings is not None:
            np.save(str(EMBEDDINGS_PATH), embeddings)
            print(f"[✓] Embeddings saved to {EMBEDDINGS_PATH}")
        
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"[✓] Metadata saved to {METADATA_PATH}")
    
    def load_index(self):
        """Load a pre-built FAISS index and metadata from disk."""
        print(f"[→] Loading FAISS index from {FAISS_INDEX_PATH}...")
        
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"    Index size: {self.index.ntotal:,} vectors")
        
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"    Metadata entries: {len(self.metadata):,}")
    
    def retrieve(self, query: str, top_k: int = None) -> list:
        """
        Retrieve the top-K most similar historical conversations.
        
        Args:
            query: The incoming customer message.
            top_k: Number of results to return (default from config).
        
        Returns:
            List of dicts with keys:
                - customer_text: The similar historical customer message
                - agent_text: How the agent responded to that message
                - similarity: Cosine similarity score (0 to 1)
                - thread_id: Source thread identifier
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")
        
        if top_k is None:
            top_k = TOP_K_RETRIEVAL
        
        # Embed the query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            results.append({
                "customer_text": meta["customer_text"],
                "agent_text": meta["agent_text"],
                "similarity": float(sim),
                "thread_id": meta["thread_id"],
            })
        
        return results
    
    def get_max_similarity(self, query: str) -> float:
        """Get the similarity score of the best match (used for escalation logic)."""
        results = self.retrieve(query, top_k=1)
        if results:
            return results[0]["similarity"]
        return 0.0
