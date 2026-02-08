import os
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_modules.engine import AdvancedTourismEngine
from rag_modules.bot import AdvancedBot
from rag_modules.llm_client import GroqClient

app = FastAPI(title="Tourism RAG API", version="1.1")

# --- CORS MIDDLEWARE (Fix for Browser Access) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for dev/demo). Lock this down in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "locations.csv")
INDICES_DIR = os.path.join(BASE_DIR, "tourism_indices")
DB_PATH = os.path.join(BASE_DIR, "tourism_advanced_db")

# Global variables
bot = None

class QueryRequest(BaseModel):
    query: str
    k: int = 5 

class QueryResponse(BaseModel):
    response: str
    context_sources: list
    timings: dict

@app.on_event("startup")
def load_resources():
    global bot
    print("Loading RAG Engine...")
    
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"Data file not found at {CSV_PATH}")
        
    engine = AdvancedTourismEngine(CSV_PATH, INDICES_DIR, DB_PATH)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not set. LLM generation will fail.")
        
    llm = GroqClient(api_key=api_key)
    bot = AdvancedBot(engine, llm)
    print("System Ready.")

# --- CONCURRENCY FIX: Removed 'async' ---
# By using standard 'def', FastAPI runs this in a thread pool, 
# preventing the synchronous LLM/Vector search from blocking other requests.
@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
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

@app.get("/sys/packages")
def get_requirements():
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    return {"packages": result.stdout.split("\n")}