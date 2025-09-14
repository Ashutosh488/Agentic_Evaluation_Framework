# ==============================================================================
# 🚀 AGENTIC EVALUATION FRAMEWORK - HUGGING FACE EDITION 🚀
# ==============================================================================
# This version uses the Hugging Face Inference API for a lightweight, deployable app.

import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🤖 Agentic Evaluation Framework",
    page_icon="🚀",
    layout="wide",
)

# --- MODEL & APP CONSTANTS ---
GENERATOR_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
JUDGE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

# --- SIDEBAR FOR API KEY ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Enter your Hugging Face API key to connect to the models.")

# Use st.secrets for deployment, with a fallback to text_input for local development
if 'HF_API_KEY' in st.secrets:
    hf_api_key = st.secrets['HF_API_KEY']
    st.sidebar.success("API key loaded from secrets!", icon="✅")
else:
    hf_api_key = st.sidebar.text_input("Hugging Face API Key", type="password")

if not hf_api_key:
    st.info("Please enter your Hugging Face API key in the sidebar to begin.")
    st.stop()

# --- CORE FUNCTIONS (LLM Logic with Hugging Face API) ---
@st.cache_data(show_spinner=False)
def query_hf_model(api_url: str, payload: dict, api_key: str) -> str:
    """Sends a request to a Hugging Face model and returns the text response."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 503: # Model is loading
            st.toast("Model is loading, please wait...")
            time.sleep(15)
            response = requests.post(api_url, headers=headers, json=payload)

        response.raise_for_status()

        result = response.json()
        generated_text = result[0].get('generated_text', '')
        
        input_text = payload.get("inputs", "")
        if generated_text.startswith(input_text):
            return generated_text[len(input_text):].strip()
        return generated_text.strip()

    except requests.exceptions.RequestException as e:
        # This will be printed in the terminal if the key is wrong or network fails
        print(f"API Request Error: {e}")
        # Return an error string that will cause a downstream failure
        return f"ERROR_API_REQUEST: {e}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing model response: {e}")
        return f"ERROR_PARSING: {response.text}"


@st.cache_data(show_spinner=False)
def evaluate_response_in_one_shot(prompt: str, response: str, metadata: str) -> dict:
    """Evaluates a response using the Judge model, returning a structured dictionary."""
    if not metadata: metadata = "No metadata provided."

    judge_system_prompt = "You are a fair AI evaluator. Your response MUST be a single, valid JSON object with the requested keys and nothing else."
    judge_user_prompt = f"""
    Evaluate the agent's response based on the prompt and metadata.
    **Metadata:** --- {metadata} ---
    **Original Prompt:** --- {prompt} ---
    **Agent's Response:** --- {response} ---
    Provide your evaluation as a JSON object with four keys: "instruction_following", "coherence", "assumption_control", and "hallucination_score", each with a "score" (int from 0 to 10) and "justification" (string).
    """
    
    full_prompt = f"<|system|>\n{judge_system_prompt}<|end|>\n<|user|>\n{judge_user_prompt}<|end|>\n<|assistant|>"

    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.1}
    }

    raw_output = query_hf_model(JUDGE_API_URL, payload, hf_api_key)

    try:
        clean_json_str = raw_output.strip().removeprefix('```json').removesuffix('```')
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        return {key: {"score": -1, "justification": f"Failed to parse LLM's JSON output. Raw: {raw_output}"} for key in ["instruction_following", "coherence", "assumption_control", "hallucination_score"]}

# --- UI HELPER FUNCTIONS ---
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
            st.metric(label="Score", value=f"{score} / 10", label_visibility="collapsed")
            st.progress(score * 10)
            with st.expander("See Justification"):
                st.info(f"_{justification}_")
        col_idx += 1

# --- MAIN APP LAYOUT ---
st.title("🚀 Agentic Evaluation Framework")
st.markdown("An interactive tool to evaluate and visualize AI agent performance using the **Hugging Face Inference API**.")

tab1, tab2 = st.tabs(["Interactive Evaluator", "Batch Analysis & Visualization"])

with tab1:
    st.header("Live Evaluation")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        prompt_input = st.text_area("1. User Prompt", height=150)
        response_input = st.text_area("2. Agent's Response", height=200)
        metadata_input = st.text_area("3. Metadata (Optional JSON)", height=100)
        if st.button("Evaluate Now ✨", type="primary", use_container_width=True):
            if not prompt_input or not response_input:
                st.error("Please provide both a prompt and a response.")
            else:
                with st.spinner("🤖 The Judge is evaluating..."):
                    st.session_state.evaluation_results = evaluate_response_in_one_shot(prompt_input, response_input, metadata_input)
    with col2:
        if 'evaluation_results' in st.session_state:
            display_evaluation_report(st.session_state.evaluation_results)

with tab2:
    st.header("Batch Analysis Pipeline")
    with st.expander("Step 1: Generate Synthetic Data", expanded=True):
        num_samples = st.slider("Number of prompt/response pairs to generate", 1, 10, 2)
        if st.button("Generate Data 📝", use_container_width=True):
            all_data = []
            progress_bar = st.progress(0, text="Generating data...")
            topics = ["Renewable Energy", "Ancient Rome", "Machine Learning Concepts"]
            for i in range(num_samples):
                topic = random.choice(topics)
                system_prompt = "You are a synthetic data generator. Your output MUST be a single, valid JSON object and nothing else."
                user_prompt = f"Create a user prompt, metadata, a high-quality 'Agent A' response, and a low-quality 'Agent B' response on the topic of **{topic}**. The JSON must have these keys: 'metadata', 'prompt', 'agent_a_response', 'agent_b_response'."
                generator_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                payload = {"inputs": generator_prompt, "parameters": {"max_new_tokens": 1024}}
                raw_output = query_hf_model(GENERATOR_API_URL, payload, hf_api_key)
                try:
                    data_point = json.loads(raw_output)
                    all_data.append({'agent_id': 'Agent-A', 'prompt': data_point.get('prompt'), 'metadata': json.dumps(data_point.get('metadata')), 'agent_response': data_point.get('agent_a_response')})
                    all_data.append({'agent_id': 'Agent-B', 'prompt': data_point.get('prompt'), 'metadata': json.dumps(data_point.get('metadata')), 'agent_response': data_point.get('agent_b_response')})
                except (json.JSONDecodeError, AttributeError) as e:
                    st.warning(f"Skipped a data point due to a generation/parsing error: {e}")
                progress_bar.progress((i + 1) / num_samples, f"Generated pair {i+1}/{num_samples}")
            st.session_state.generated_df = pd.DataFrame(all_data)
            progress_bar.empty()
            st.success(f"✅ Generated {len(st.session_state.generated_df)} rows.")

    if 'generated_df' in st.session_state and not st.session_state.generated_df.empty:
        st.dataframe(st.session_state.generated_df)

    st.divider()
    st.subheader("Step 2: Run Evaluation and Visualize Results")
    if 'generated_df' in st.session_state and not st.session_state.generated_df.empty:
        if st.button("Run Full Analysis 📊", type="primary", use_container_width=True):
            with st.spinner("Running batch evaluation..."):
                df = st.session_state.generated_df
                results = []
                progress_bar = st.progress(0, text="Evaluating responses...")
                for index, row in df.iterrows():
                    result_dict = evaluate_response_in_one_shot(row.get('prompt'), row.get('agent_response'), row.get('metadata'))
                    results.append(result_dict)
                    progress_bar.progress((index + 1) / len(df), f"Evaluating response {index+1}/{len(df)}")
                results_df = pd.json_normalize(results, sep='_')
                evaluated_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                score_cols = [col for col in evaluated_df.columns if col.endswith('_score')]
                for col in score_cols:
                    evaluated_df[f'{col}_int'] = pd.to_numeric(evaluated_df[col], errors='coerce').fillna(0)
                int_score_cols = [col for col in evaluated_df.columns if '_int' in col]
                evaluated_df['average_score'] = evaluated_df[int_score_cols].mean(axis=1)
                st.session_state.analyzed_df = evaluated_df
            st.success("✅ Batch analysis complete!")

    if 'analyzed_df' in st.session_state:
        st.subheader("Analysis Results")
        analyzed_df = st.session_state.analyzed_df
        # Visualization logic...
