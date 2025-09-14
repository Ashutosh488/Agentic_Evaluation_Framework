# api_wrapper.py
from evaluator import evaluate_response
import json

def run_evaluation(prompt: str, response: str, metadata: str):
    """
    Calls evaluator.evaluate_response(prompt, response, metadata)
    Returns:
      - scores: normalized floats (0.0 - 1.0)
      - explanation_text: combined justifications (string)
      - raw_details: original LLM JSON (dict) so frontend can show per-dimension justifications
    """
    results = evaluate_response(prompt, response, metadata)
    # Defensive get — in case evaluator returns missing keys
    def get_score(key, fallback=0):
        return results.get(key, {}).get("score", fallback)

    scores = {
        "instruction_following": get_score("instruction_following") / 10.0,
        "coherence": get_score("coherence") / 10.0,
        "assumption_control": get_score("assumption_control") / 10.0,
        # NOTE: evaluator uses "hallucination_score" key — map it:
        "hallucination": get_score("hallucination_score") / 10.0
    }

    # Build an explanation text (human friendly)
    explanation_lines = []
    for k, v in results.items():
        justification = v.get("justification", "")
        score = v.get("score", "")
        explanation_lines.append(f"{k}: (score={score}) {justification}")
    explanation_text = "\n".join(explanation_lines)

    return scores, explanation_text, results
