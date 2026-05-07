"""
ingest.py — Load a PDF, split into chunks, create embeddings, store in FAISS.

Embeddings strategy:
    HuggingFace sentence-transformers

"""

import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
load_dotenv()


UPLOAD_DIR    = "data"
FAISS_INDEX   = "faiss_index"

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)




# ── Embedding selection 

def get_embeddings():
    """
    Returns an embedding model.

    I used an hugging face fallback because it is free, runs on the pc itself and downloads on the first try itself. 
    the size is around 90mb and even though it is not as clear as an api embedding model, 
    the difference isn't drastic for a dataset this small
    """

    print("USING HuggingFace embeddings since Azure AI embeddings is not present")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Main pipeline 

def run_ingestion(pdf_path: str) -> dict:
    """

    Returns a summary dict.
    """
    # 1. Load
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
 
    if not documents:
        raise ValueError("PDF appears to be empty or is image-based (scanned). Text extraction failed.")
 
    # 2. Split
    splitter = SemanticChunker(
    get_embeddings(),
    breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_documents(documents)
 
    if not chunks:
        raise ValueError("No chunks were created. Check if the PDF contains extractable text.")
 
    # 3. Embed + 4. Store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX)
 
    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "faiss_index": FAISS_INDEX,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Ingestion API ready.")
    yield
    print("Ingestion API shutting down...")
 
 
app = FastAPI(
    title="RAG Ingestion API",
    description="Upload a PDF → chunk → embed → store in FAISS",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# Schemas 
 
class StatusResponse(BaseModel):
    ready: bool
    message: str
    faiss_index: str
 
class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int
    faiss_index: str
 
 

 
@app.get("/status", response_model=StatusResponse, summary="Check if FAISS index is ready")
def status():
    """
    Returns whether a FAISS index exists on disk.
    """
    index_exists = (
        Path(f"{FAISS_INDEX}/index.faiss").exists() and
        Path(f"{FAISS_INDEX}/index.pkl").exists()
    )
    return StatusResponse(
        ready=index_exists,
        message="FAISS index is ready." if index_exists else "No FAISS index found. Upload a PDF first.",
        faiss_index=FAISS_INDEX,
    )
 
 
@app.post("/upload", response_model=UploadResponse, summary="Upload a PDF and run ingestion")
async def upload(file: UploadFile = File(...)):
    """
    Accepts a pdf file and the chat api will use the updated index
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
 
    # Save uploaded file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()
 
    # Run ingestion pipeline
    try:
        result = run_ingestion(save_path)
    except ValueError as e:
        # Clean up saved file if ingestion fails
        Path(save_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        Path(save_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
 
    return UploadResponse(
        message="PDF ingested successfully. You can now use the chat API.",
        filename=file.filename,
        pages=result["pages"],
        chunks=result["chunks"],
        faiss_index=result["faiss_index"],
    )
 