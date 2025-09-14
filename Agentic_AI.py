# ==============================================================================
# 🚀 AGENTIC EVALUATION FRAMEWORK - UNIVERSAL SUBMISSION CODE 🚀
# e6data x IIT BHU Hackathon 2025 | Problem P4: Agentic Evaluation
#
# This script is universal and works on both Google Colab and local machines.
# - On Colab: It will automatically install and start the Ollama server.
# - On Local: It assumes you have manually installed and started the Ollama server.
# ==============================================================================

import sys
import os
import time
import json
import subprocess
import pandas as pd
import ollama
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import ipywidgets as widgets

# --- STEP 1: ENVIRONMENT-SPECIFIC SETUP ---
print("--- ⚙️ STEP 1: CONFIGURING THE ENVIRONMENT ---")

# --- UNIVERSAL SETUP: Detect environment and adapt ---
IN_COLAB = 'google.colab' in sys.modules

def run_command(command):
    """Helper function to run shell commands and handle errors."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode('utf-8'))
    return stdout.decode('utf-8')

if IN_COLAB:
    print("✅ Google Colab environment detected. Automating Ollama setup...")
    run_command(f"{sys.executable} -m pip install -q ollama")
    run_command("curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1")
    os.environ['OLLAMA_HOST'] = '0.0.0.0'
    run_command("nohup ollama serve &> ollama.log &")
    time.sleep(5) # Wait for server to initialize
    print("✅ Ollama server started on Colab.")
else:
    print("✅ Local environment detected. Please ensure the Ollama server is running manually.")
    print("   (Run 'ollama serve' in your terminal).")

print("\nDownloading AI models (this may take a few minutes)...")
try:
    ollama.pull('phi3:mini')
    ollama.pull('llama3:8b')
    print("✅ Models are ready.")
except Exception as e:
    print(f"❌ Could not connect to Ollama to pull models. Please ensure the Ollama application is running. Error: {e}")
    # Exit gracefully if Ollama isn't running, to avoid further errors.
    sys.exit()


# --- STEP 2: DEFINE CORE FUNCTIONS ---
print("\n--- 🛠️ STEP 2: INITIALIZING EVALUATION FRAMEWORK ---")

JUDGE_MODEL = "phi3:mini"

def get_llm_judgment(prompt: str) -> str:
    """Sends a prompt to the local LLM judge."""
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
            format="json"
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return '{"error": "Failed to get response from Ollama server."}'

def evaluate_response(prompt: str, response: str, metadata: str) -> dict:
    """Evaluates a single agent response using the AI judge."""
    if not metadata or pd.isna(metadata): metadata = "No metadata provided."
    judge_prompt = f"""
    You are a fair AI evaluator. Evaluate the agent's response based on the prompt and metadata.
    **Metadata:** --- {metadata} ---
    **Original Prompt:** --- {prompt} ---
    **Agent's Response:** --- {response} ---
    Your response MUST be a single, valid JSON object with four keys: "instruction_following", "coherence", "assumption_control", and "hallucination_score".
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


# --- STEP 3: GENERATE & PROCESS DATASET (DEMONSTRATES SCALABILITY) ---
print("\n--- 📝 STEP 3: GENERATING AND PROCESSING DATASET ---")

def generate_synthetic_data(filepath="synthetic_agent_data.csv", num_rows=50): # Reduced for faster demo
    """Generates a CSV file to demonstrate batch processing."""
    if os.path.exists(filepath):
        print(f"✅ Found existing dataset at '{filepath}'.")
        return
    print(f"Generating a new synthetic dataset with {num_rows} entries...")
    data = []
    # ... (Code to generate data as in the previous response) ...
    for i in range(num_rows):
        if i % 4 == 0:
            data.append({"agent_id": "Agent-A (High Quality)", "prompt": 'List three colors in a JSON array with the key "colors".', "metadata": '{"task": "json_output"}', "agent_response": '{"colors": ["red", "green", "blue"]}'})
        elif i % 4 == 1:
            data.append({"agent_id": "Agent-B (Low Quality)", "prompt": 'List three colors in a JSON array with the key "colors".', "metadata": '{"task": "json_output"}', "agent_response": 'Here are the colors: red, green, blue.'})
        elif i % 4 == 2:
            data.append({"agent_id": "Agent-A (High Quality)", "prompt": "Summarize Newton's First Law in one sentence.", "metadata": '{"domain": "physics"}', "agent_response": "An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force."})
        else:
            data.append({"agent_id": "Agent-B (Low Quality)", "prompt": "What is the capital of the moon?", "metadata": '{"domain": "astronomy"}', "agent_response": "The capital of the moon is Tranquility Base."})
    pd.DataFrame(data).to_csv(filepath, index=False)
    print(f"✅ Synthetic dataset saved to '{filepath}'.")

DATA_FILE = "synthetic_agent_data.csv"
generate_synthetic_data(DATA_FILE)

print("\n--- 🔄 STEP 4: RUNNING BATCH EVALUATION ---")
df = pd.read_csv(DATA_FILE)
results_cache_file = "evaluation_results.csv"

if os.path.exists(results_cache_file):
    print("✅ Found cached evaluation results. Loading instantly.")
    df_final_report = pd.read_csv(results_cache_file)
else:
    print("No cache found. Evaluating all responses (this may take a few minutes)...")
    evaluation_results = []
    start_time = time.time()
    for index, row in df.iterrows():
        print(f"  - Evaluating response {index + 1}/{len(df)}...")
        eval_result = evaluate_response(row['prompt'], row['agent_response'], row['metadata'])
        flat_result = {
            'agent_id': row['agent_id'], 'prompt': row['prompt'], 'metadata': row['metadata'], 'agent_response': row['agent_response'],
            'instruction_following_score': eval_result['instruction_following']['score'], 'instruction_following_justification': eval_result['instruction_following']['justification'],
            'coherence_score': eval_result['coherence']['score'], 'coherence_justification': eval_result['coherence']['justification'],
            'assumption_control_score': eval_result['assumption_control']['score'], 'assumption_control_justification': eval_result['assumption_control']['justification'],
            'hallucination_score': eval_result['hallucination_score']['score'], 'hallucination_score_justification': eval_result['hallucination_score']['justification'],
        }
        evaluation_results.append(flat_result)
    end_time = time.time()
    print(f"✅ Batch evaluation complete in {end_time - start_time:.2f} seconds.")
    df_final_report = pd.DataFrame(evaluation_results)
    df_final_report.to_csv(results_cache_file, index=False)
    print(f"✅ Evaluation results saved to '{results_cache_file}'.")

# --- STEP 5: ANALYSIS & VISUALIZATION ---
print("\n--- 📈 STEP 5: GENERATING DASHBOARD VISUALIZATIONS ---")
# ... (Visualization code from previous response, it is unchanged) ...
def analyze_and_visualize(df: pd.DataFrame):
    score_cols = [col for col in df.columns if col.endswith('_score')]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['average_score'] = df[score_cols].mean(axis=1)
    leaderboard = df.groupby('agent_id')['average_score'].mean().sort_values(ascending=False).reset_index()
    sns.set_theme(style="whitegrid")
    print("\n--- 🏆 Agent Performance Leaderboard ---")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=leaderboard, x='average_score', y='agent_id', palette='viridis', hue='agent_id', dodge=False, legend=False)
    plt.title('Overall Agent Performance', fontsize=16)
    plt.xlabel('Average Score (out of 10)')
    plt.ylabel('Agent ID')
    plt.xlim(0, 10)
    plt.show()
    print("\n--- 🔥 Strengths & Weaknesses Heatmap ---")
    heatmap_data = df.groupby('agent_id')[score_cols].mean()
    heatmap_data.columns = [col.replace('_score', '').replace('_', ' ').title() for col in heatmap_data.columns]
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title('Average Scores by Dimension', fontsize=16)
    plt.show()

analyze_and_visualize(df_final_report.copy())
print("✅ Dashboard generated.")

# --- STEP 6: INTERACTIVE DEMO UI (LIVE) ---
print("\n--- 🚀 STEP 6: LAUNCHING LIVE INTERACTIVE DEMO ---")
# ... (UI code from previous response, it is unchanged) ...
prompt_input_ui = widgets.Textarea(placeholder='Enter the original prompt...', layout={'width': '100%', 'height': '100px'})
response_input_ui = widgets.Textarea(placeholder="Enter the agent's response...", layout={'width': '100%', 'height': '100px'})
metadata_input_ui = widgets.Textarea(placeholder='(Optional) Enter metadata as JSON...', layout={'width': '100%', 'height': '80px'})
eval_button_ui = widgets.Button(description="Evaluate", button_style='success', layout={'width': '100%'}, icon='play')
output_area_ui = widgets.Output()

def display_rich_report(results_dict):
    report_widgets = []
    total_score, num_scores = 0, 0
    for key, value in results_dict.items():
        if isinstance(value, dict):
            title, score, justification = key.replace('_', ' ').title(), int(value.get('score', 0)), value.get('justification', 'N/A')
            total_score += score
            num_scores += 1
            report_widgets.extend([widgets.Label(title), widgets.IntProgress(value=score, min=0, max=10, description=f'{score}/10', bar_style='info' if score >= 5 else 'warning'), widgets.HTML(f"<i>“{justification}”</i>")])
    if num_scores > 0:
        report_widgets.insert(0, widgets.HTML(f"<h4>Overall Score: {total_score / num_scores:.2f} / 10</h4>"))
    with output_area_ui:
        display(widgets.VBox(report_widgets))

def on_button_clicked_ui(b):
    output_area_ui.clear_output()
    prompt, response, metadata = prompt_input_ui.value, response_input_ui.value, metadata_input_ui.value
    with output_area_ui:
        if not prompt or not response:
            display(HTML("<b style='color:red;'>❌ Please enter both a prompt and a response.</b>"))
            return
        display(HTML("<i>🤔 Evaluating live with AI judge...</i>"))
        results_dict = evaluate_response(prompt, response, metadata)
        output_area_ui.clear_output()
        display_rich_report(results_dict)
eval_button_ui.on_click(on_button_clicked_ui)
input_box = widgets.VBox([widgets.Label("1. Prompt:"), prompt_input_ui, widgets.Label("2. Agent's Response:"), response_input_ui, widgets.Label("3. Metadata (Context):"), metadata_input_ui, eval_button_ui], layout={'width': '50%'})
output_box = widgets.VBox([widgets.HTML("<h3>🏆 Live Evaluation Report 🏆</h3>"), output_area_ui], layout={'width': '50%'})
app_layout = widgets.HBox([input_box, output_box])
display(HTML("<h2>Agentic Evaluation Framework: Demo</h2>"))
display(app_layout)