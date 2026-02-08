import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_modules.engine import AdvancedTourismEngine
from rag_modules.bot import AdvancedBot
from rag_modules.llm_client import GroqClient

app = FastAPI(title="Tourism RAG API", version="1.0")

# --- Config ---
# Paths inside the Docker container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "locations.csv")
INDICES_DIR = os.path.join(BASE_DIR, "tourism_indices")
DB_PATH = os.path.join(BASE_DIR, "tourism_advanced_db")

# Global variables
bot = None

class QueryRequest(BaseModel):
    query: str
    k: int = 5  # Default to 5, but user can change it

class QueryResponse(BaseModel):
    response: str
    context_sources: list
    timings: dict

@app.on_event("startup")
def load_resources():
    global bot
    print("Loading RAG Engine...")
    
    # 1. Initialize Engine
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"Data file not found at {CSV_PATH}")
        
    engine = AdvancedTourismEngine(CSV_PATH, INDICES_DIR, DB_PATH)
    
    # 2. Initialize LLM (Swap this class to change providers)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not set. LLM generation will fail.")
        
    llm = GroqClient(api_key=api_key)
    
    # 3. Initialize Bot
    bot = AdvancedBot(engine, llm)
    print("System Ready.")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not bot:
        raise HTTPException(status_code=503, detail="System starting up")
    
    try:
        result = bot.process_query(request.query, k=request.k)
        return QueryResponse(
            response=result["response"],
            context_sources=result["context"],
            timings=result["timings"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "active", "docs_url": "/docs"}