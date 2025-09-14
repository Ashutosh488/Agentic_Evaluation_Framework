# server.py
import json
import uvicorn
from typing import Optional, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from api_wrapper import run_evaluation
import pandas as pd

app = FastAPI(title="Agentic Eval API")

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvalRequest(BaseModel):
    prompt: str
    response: str
    metadata: Optional[Any] = None

@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    print("Incoming request:", req.dict())
    # Normalize metadata to string for the evaluator (it expects text)
    if isinstance(req.metadata, dict):
        metadata_str = json.dumps(req.metadata)
    elif req.metadata is None:
        metadata_str = ""
    else:
        metadata_str = str(req.metadata)

    try:
        scores, explanation_text, raw_details = run_evaluation(
            req.prompt, req.response, metadata_str
        )
        response = {
            "scores": scores,
            "explanation": explanation_text,
            "details": raw_details,
        }
        print("Outgoing response:", response)
        return response
    except Exception as e:
        print("Evaluation error:", e)
        return {"error": str(e)}


# ---------------------------
# 🚀 New: Batch evaluation
# ---------------------------

class EvalBatchRequest(BaseModel):
    agents: List[EvalRequest]

@app.post("/evaluate_batch")
async def evaluate_batch(req: EvalBatchRequest):
    results = []
    for agent in req.agents:
        metadata_str = (
            json.dumps(agent.metadata) if isinstance(agent.metadata, dict) else str(agent.metadata or "")
        )
        scores, explanation, details = run_evaluation(agent.prompt, agent.response, metadata_str)

        agent_id = (
            agent.metadata.get("agent_id", "Unknown")
            if isinstance(agent.metadata, dict)
            else "Unknown"
        )

        results.append({
            "agent_id": agent_id,
            "prompt": agent.prompt,
            "response": agent.response,
            "metadata": agent.metadata,
            "scores": scores,
            "details": details,
            "explanation": explanation,
        })

    # Build leaderboard dataframe
    df = pd.DataFrame([{
        "agent_id": r["agent_id"],
        "instruction_following": r["scores"]["instruction_following"],
        "coherence": r["scores"]["coherence"],
        "assumption_control": r["scores"]["assumption_control"],
        "hallucination": r["scores"]["hallucination"]
    } for r in results])

    leaderboard = df.groupby("agent_id").mean().reset_index()

    return {
        "results": results,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
