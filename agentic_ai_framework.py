# ==============================================================================
# 🚀 AGENTIC EVALUATION FRAMEWORK - OPENAI EDITION 🚀
# ==============================================================================
# This version uses the OpenAI API for robust, reliable evaluation.

import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from openai import OpenAI, AuthenticationError, APIError

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🤖 Agentic Evaluation Framework (OpenAI)",
    page_icon="🚀",
    layout="wide",
)

# --- MODEL & APP CONSTANTS ---
# Using gpt-4o-mini for its speed, cost-effectiveness, and strong JSON capabilities.
GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# --- SIDEBAR FOR API KEY ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Enter your OpenAI API key to connect to the models.")

# Use st.secrets for deployment, with a fallback to text_input for local development
if 'OPENAI_API_KEY' in st.secrets:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    st.sidebar.success("API key loaded from secrets!", icon="✅")
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=openai_api_key)
except AuthenticationError:
    st.error("The provided OpenAI API key is invalid. Please check and try again.")
    st.stop()

# --- CORE FUNCTIONS (LLM Logic with OpenAI API) ---
@st.cache_data(show_spinner="Connecting to OpenAI model...")
def query_openai_model(model: str, messages: list, expect_json: bool = False) -> str:
    """Sends a request to an OpenAI model and returns the text response."""
    try:
        response_format = {"type": "json_object"} if expect_json else {"type": "text"}
        
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format=response_format,
        )
        return chat_completion.choices[0].message.content
    except APIError as e:
        raise ConnectionError(f"OpenAI API Error: {e.body.get('message', 'Unknown error')}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

@st.cache_data(show_spinner="The AI Judge is evaluating...")
def evaluate_response_in_one_shot(prompt: str, response: str, metadata: str) -> dict:
    """Evaluates a response using the Judge model, returning a structured dictionary."""
    if not metadata: metadata = "No metadata provided."

    system_prompt = "You are a fair AI evaluator. Your response MUST be a single, valid JSON object with the requested keys and nothing else."
    user_prompt = f"""
    Evaluate the agent's response based on the prompt and metadata.
    **Metadata:** --- {metadata} ---
    **Original Prompt:** --- {prompt} ---
    **Agent's Response:** --- {response} ---
    Provide your evaluation as a JSON object with four keys: "instruction_following", "coherence", "assumption_control", and "hallucination_score", each with a "score" (int from 0 to 10) and "justification" (string).
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Using expect_json=True for reliable JSON output
        raw_output = query_openai_model(JUDGE_MODEL, messages, expect_json=True)
        return json.loads(raw_output)
    except (json.JSONDecodeError, ConnectionError, ValueError) as e:
        return {key: {"score": -1, "justification": f"Evaluation failed. Error: {e}"} for key in ["instruction_following", "coherence", "assumption_control", "hallucination_score"]}

# --- UI HELPER FUNCTIONS (Unchanged) ---
def display_evaluation_report(results_dict: dict):
    """Renders a visually appealing report in Streamlit."""
    st.subheader("🏆 Evaluation Report")
    if not results_dict or results_dict.get('instruction_following', {}).get('score', -1) == -1:
        st.error(f"Could not generate a report. Justification: {results_dict.get('instruction_following', {}).get('justification', 'Unknown error')}")
        return

    scores = {key: value.get('score', 0) for key, value in results_dict.items()}
    avg_score = sum(scores.values()) / len(scores) if scores else 0

    st.metric(label="**Average Score**", value=f"{avg_score:.2f} / 10")
    st.progress(int(avg_score * 10))
    st.divider()

    cols = st.columns(2)
    col_idx = 0
    for key, value in results_dict.items():
        with cols[col_idx % 2]:
            title = key.replace('_', ' ').title()
            score = value.get('score', 0)
            justification = value.get('justification', 'N/A')

            st.markdown(f"**{title}**")
            if score >= 8: color = "blue"
            elif score >= 5: color = "orange"
            else: color = "red"
            st.markdown(f'<div style="width: 100%; background-color: #eee; border-radius: 5px;"><div style="width: {score*10}%; background-color: {color}; color: white; text-align: center; border-radius: 5px;">{score}/10</div></div>', unsafe_allow_html=True)
            with st.expander("See Justification"):
                st.info(f"_{justification}_")
        col_idx += 1

# --- MAIN APP LAYOUT (Unchanged) ---
st.title("🚀 Agentic Evaluation Framework")
st.markdown("An interactive tool to evaluate and visualize AI agent performance using the **OpenAI API**.")

tab1, tab2 = st.tabs(["Interactive Evaluator", "Batch Analysis & Visualization"])

with tab1:
    st.header("Live Evaluation")
