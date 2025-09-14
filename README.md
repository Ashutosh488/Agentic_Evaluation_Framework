# 🤖 Agentic Evaluation Framework  
A Project for the e6data x IIT BHU Hackathon 2025 by **The Pareto Crew**  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)](https://colab.research.google.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## 🚩 The Problem: The Scaling Challenge of AI Evaluation  

As AI agents become more powerful and widespread, *evaluating their performance* is a critical bottleneck.  

Manual evaluation is:  
- ❌ Slow  
- ❌ Expensive  
- ❌ Subjective  
- ❌ Impossible to scale  

When a company has hundreds of agents producing thousands of responses, how can they efficiently ensure that the agents:  
- ✅ Follow instructions precisely  
- ✅ Avoid factual hallucinations  
- ✅ Provide coherent and useful responses  
- ✅ Do not make unwarranted assumptions  

Without an automated and robust solution, *building trustworthy AI is not feasible*.  

---

## 💡 Our Solution: An Automated Evaluation & Comparison Platform  

We developed the *Agentic Evaluation Framework, an **interactive platform* designed to solve the problem of large-scale agent evaluation.  

It provides developers and researchers with tools to:  
- 🔍 Automatically score agent responses  
- ⚖ Compare different models head-to-head  
- 📈 Track performance over time  

Our framework directly addresses the hackathon challenge by accepting prompts, responses, and metadata to output an *interpretable performance report*.  

---

## ✨ Key Features  

Built as an *interactive Google Colab notebook*, our framework includes:  

- ⚡ *Batch Processing Engine*  
  Process thousands of responses from a CSV file, scoring each one across four key dimensions:  
  Instruction-following, coherence, assumption control, hallucination detection.  

- 🎯 *Live "Head-to-Head" Demo*  
  Input a single prompt → two AI models generate responses → both evaluated & displayed *side-by-side in real-time*.  

- 📊 *Performance Dashboard*  
  A visualization tab with a *ranked leaderboard* of evaluated agents, offering a clear overview of performance.  

- 🧩 *Metadata Integration*  
  Evaluation process accepts metadata for *context-aware judgments*.  

---

## 🛠 Technical Methodology  

- *Core Technology:* Python (Google Colab, T4 GPU)  
- *Local LLMs with Ollama:* llama3:8b and phi3:mini (open-source, zero cost, highly flexible)  
- *Asynchronous Processing:* asyncio executes generation + evaluation calls in parallel → *faster results*  
- *Interactive UI:* Built with ipywidgets for a *responsive Colab experience*  

---

## ⚙ Setup & Usage Instructions  

### 🔧 1. Open in Google Colab  
- Open Agentic_AI.py in Colab  
- Copy-paste code into a single cell  

### ⚙ 2. Set the Runtime  
- Go to Runtime → Change runtime type  
- Select *T4 GPU*  

### 📦 3. Run Setup  
- Run *Cell 1*  
- Installs Ollama, pulls required models, installs dependencies  
- Takes a few minutes  

### 📂 4. Upload Data  
- Use Colab file browser (left panel)  
- Upload final_report_with_scores.csv  
- Required for Performance Dashboard  

### 🚀 5. Launch the App  
- Run *Cell 2*  
- Opens interactive *Head-to-Head Demo* + *Performance Dashboard*  

---

## 📊 Example Workflow  

1. Upload a dataset with agent responses  
2. Run *batch evaluation* → get scores across 4 metrics  
3. View *Performance Dashboard* → ranked leaderboard  
4. Use *Head-to-Head Demo* → real-time model comparison  

---

## 📜 License  

This project is licensed under the [MIT License](LICENSE).  

---

## 🙌 Acknowledgements  

Developed by *The Pareto Crew* for the e6data x IIT BHU Hackathon 2025.