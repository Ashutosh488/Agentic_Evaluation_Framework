# evaluator.py
import json
import pandas as pd
import ollama

JUDGE_MODEL = "phi3:mini"

def get_llm_judgment(prompt: str) -> str:
    """Sends a prompt to the local LLM judge."""
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json"
    )
    return response['message']['content']

def evaluate_response(prompt: str, response: str, metadata: str) -> dict:
    """Evaluates a single agent response using the AI judge."""
    if not metadata or pd.isna(metadata): 
        metadata = "No metadata provided."
    
    judge_prompt = f"""
    You are a fair AI evaluator. Evaluate the agent's response based on the prompt and metadata.
    **Metadata:** --- {metadata} ---
    **Original Prompt:** --- {prompt} ---
    **Agent's Response:** --- {response} ---
    Your response MUST be a single, valid JSON object with four keys: "instruction_following", 
    "coherence", "assumption_control", and "hallucination_score".
    Each key must have a nested JSON object with an integer "score" (0-10) and a "justification" string.
    """
    
    raw_output = get_llm_judgment(judge_prompt)
    try:
        results = json.loads(raw_output)
        for key in ["instruction_following", "coherence", "assumption_control", "hallucination_score"]:
            if key not in results:
                results[key] = {"score": 0, "justification": f"Missing key '{key}' in LLM response."}
        return results
    except json.JSONDecodeError:
        return {key: {"score": 0, "justification": "Fatal Error: LLM judge returned invalid JSON."} for key in ["instruction_following", "coherence", "assumption_control", "hallucination_score"]}
