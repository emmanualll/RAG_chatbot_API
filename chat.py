"""
chat.py is ran after ingest.py
it uses the FAISS index in order to retrieve releveant chunks of data and answer via the AzureOpenAI

this uses the module from langchain_openai
"""

import os
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

#for embeddings and retreving data including the models
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

#fastapi modules
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

#configuration
FAISS_INDEX           = "faiss_index"
AZURE_ENDPOINT        = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY         = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION     = os.getenv("OPENAI_API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")
TOP_K                 = 4

#THE PROMPT
RAG_PROMPT = PromptTemplate(
    input_variable = ["context", "question"],
    template = """nswer using ONLY the provided context.
If the answer is not present, say:
"I don't have enough information in the provided text."

Keep the answer concise.

Context:
{context}

Question: {question}
Answer:
    """
)

def get_embeddings():
    print("Embedding again, ensuring that it matches ingest.py")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

class AppState:
    chain: ConversationalRetrievalChain = None
    memory: ConversationBufferWindowMemory = None
 
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs on startup - it laods the faiss and buils the chain once
    Fastapi ensure that this is alive for all request i.e., there is no reloading per request
    """
    print("Starting up! Loading the FAISS index and building chain...")

    for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]:
        if not os.getenv(var):
            print(f"{var} not set. Please check the env files")
            sys.exit(1)

    if not os.path.exists(FAISS_INDEX):
        print(f"FAISS index not found at '{FAISS_INDEX}/'")
        print("Please Run: python ingest.py")
        sys.exit(1)

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
        
    #azure openai llm (azure deployment is the name of the deployment in azure portal)
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        openai_api_version=AZURE_API_VERSION,
        temperature=0.2,
        max_tokens=1024,
    )

    #memory - keeps the last 5 tuens so follow up questions work
    app_state.memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    #i used a converstationalRetrievachain because it is smarter than a Retrieval QA
    #it rewrites the folowup questions using chat history
    #before hitting FAISS so that the explain furthewr thing actually works

    app_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=app_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
    )

    print(f"Ready -> Deployment: {AZURE_CHAT_DEPLOYMENT}")
    yield

    print("Shutting Down...")


#FASTAPI app
app = FastAPI(
    title = "RAG Chatbot API",
    description="Azure OpenAI + FAISS + LAngchain -- RAG over a textbook PDF",
    version= "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Schemas

class ChatRequest(BaseModel):
    question : str

    class Config:
        json_schema_extra = {
            "example": {"question": "What is multi-head attention?"}
        }
 
class SourceDocument(BaseModel):
    page: int
    preview: str
 
class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]

#endpoints
@app.get("/health", summary = "Health check")
def health():
    """Check the sever staus and whether the FAISS index is loaded """
    return {
        "status": "ok",
        "faiss_loaded": app_state.chain is not None,
        "deployment" : AZURE_CHAT_DEPLOYMENT,
    }

@app.post("/chat", response_model=ChatResponse, summary="Ask a question")
async def chat(request: ChatRequest):
    """
    This method is used to send a question, get an answer with the source page references.
    It Maintains conversational memory across calls (Last 5 turns)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
 
    if app_state.chain is None:
        raise HTTPException(status_code=503, detail="Chain not initialized.")
 
    try:
        result = app_state.chain.invoke({"question": request.question})
    except Exception as e:
        error = str(e)
        if "DeploymentNotFound" in error or "404" in error:
            raise HTTPException(status_code=502, detail=f"Azure deployment not found: {AZURE_CHAT_DEPLOYMENT}")
        if "AuthenticationError" in error or "401" in error:
            raise HTTPException(status_code=502, detail="Azure API key invalid.")
        raise HTTPException(status_code=500, detail=error)
 
    # Deduplicate sources by page number
    sources = []
    seen_pages = set()
    for doc in result.get("source_documents", []):
        page = doc.metadata.get("page", 0) + 1  # PyPDF is 0-indexed
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append(SourceDocument(
                page=page,
                preview=doc.page_content[:200].replace("\n", " "),
            ))
 
    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        sources=sources,
    )

@app.delete("/history", summary="Clear conversation memory")
def clear_history():
    """Reset conversation memory without restarting the server."""
    if app_state.memory:
        app_state.memory.clear()
    return {"status": "ok", "message": "Conversation history cleared."}