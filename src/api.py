from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
from pathlib import Path
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AmazonHelpAgent

app = FastAPI(title="AmazonHelp AI Agent Dashboard")

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        print("[→] Initializing AmazonHelp Agent...")
        agent = AmazonHelpAgent(use_fast_model=False)
        print("[✓] Agent initialized successfully!")
    except Exception as e:
        print(f"[!] Failed to initialize agent: {e}")
        traceback.print_exc()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    thread_length: int = 1

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")
    
    try:
        # Process message through our pipeline
        result = agent.process_message(
            customer_message=request.message,
            thread_length=request.thread_length
        )
        return result
    except Exception as e:
        print(f"[!] Error processing message: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def analytics_endpoint():
    try:
        eval_path = PROJECT_ROOT / "reports" / "evaluation_results.json"
        if eval_path.exists():
            import json
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Evaluation results not found. Run evaluation first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def settings_endpoint():
    from src import config
    # Return a safe subset of configuration
    return {
        "LLM_MODEL_PRIMARY": config.LLM_MODEL_PRIMARY,
        "LLM_MODEL_FAST": config.LLM_MODEL_FAST,
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        "ESCALATE_THRESHOLD": config.ESCALATE_THRESHOLD,
        "RETRIEVAL_THRESHOLD": config.RETRIEVAL_THRESHOLD,
        "MAX_RETRIEVED_EXAMPLES": config.MAX_RETRIEVED_EXAMPLES,
        "BRAND_NAME": config.BRAND_NAME,
        "GROQ_DELAY_SECONDS": config.GROQ_DELAY_SECONDS
    }

# Mount static files
static_dir = PROJECT_ROOT / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

# Mount the static directory to serve CSS and JS
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Index.html not found!</h1>"
