"""
ingest.py — Load a PDF, split into chunks, create embeddings, store in FAISS.

Run this ONCE (or whenever your PDF changes):
    python ingest.py

Embeddings strategy:
  - PRIMARY:  Azure OpenAI embeddings (if you have a text-embedding deployment)
  - FALLBACK: HuggingFace sentence-transformers (free, runs locally, no API needed)

In Azure OpenAI, embeddings use a SEPARATE deployment from chat.
Common embedding model to deploy: text-embedding-ada-002 or text-embedding-3-small
If you haven't deployed one, the fallback works perfectly for a POC.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

load_dotenv()


PDF_PATH       = "data/textbook.pdf"
FAISS_INDEX    = "faiss_index"
CHUNK_SIZE     = 1000   # characters per chunk
CHUNK_OVERLAP  = 150    # overlap between chunks to preserve context

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")




# ── Embedding selection ───────────────────────────────────────────────────────

def get_embeddings():
    """
    Returns an embedding model.

    I used an hugging face fallback because it is free, runs on the pc itself and downloads on the first try itself. 
    the size is around 90mb and even though it is not as clear as an embedding model, 
    the difference isn't drastic for a dataset this small
    """

    print("USING HuggingFace embeddings since Azure AI embeddings is not present")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest():
    # 1. Validate inputs
    if not Path(PDF_PATH).exists():
        print(f" PDF not found at: {PDF_PATH}")
        print("   Create the data/ folder and place your PDF there.")
        sys.exit(1)

    if not AZURE_API_KEY:
        print("AZURE_OPENAI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    # 2. Load PDF
    print(f"Loading PDF: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages.")

    # 3. Split into chunks
    print(f"\nSplitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Tries to split on these boundaries in order — preserves paragraph structure
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    if len(chunks) == 0:
        print(" No chunks were created. Is the PDF text-based (not scanned)?")
        sys.exit(1)

    # 4. Create embeddings
    print("\nLoading embedding model...")
    embeddings = get_embeddings()

    # 5. Store in FAISS
    print(f"\nCreating FAISS index from {len(chunks)} chunks...")
    print("   This may take a minute for large PDFs...\n")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk so chat.py can load it without re-embedding
    vectorstore.save_local(FAISS_INDEX)
    print(f"FAISS index saved to: {FAISS_INDEX}/")
    print("File saved with the names: index.faiss and index.pkl")
    print("\nNext step → run: python chat.py")


if __name__ == "__main__":
    ingest()