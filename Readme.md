# Agentic Evaluation Framework  
*A Project for the e6data x IIT BHU Hackathon 2025 by The Pareto Crew*

---

## The Problem: The Scaling Challenge of AI Evaluation
As AI agents become more powerful and widespread, evaluating their performance is a critical bottleneck. Manual evaluation is:
- Slow  
- Expensive  
- Subjective  
- Impossible to scale  

When a company has hundreds of agents producing thousands of responses, how can they efficiently ensure that the agents:  
- Follow instructions precisely?  
- Avoid factual hallucinations?  
- Provide coherent and useful responses?  
- Do not make unwarranted assumptions?  

Without an automated and robust solution, building trustworthy AI is not feasible.

---

## Our Solution: An Automated Evaluation & Comparison Platform
We have developed the **Agentic Evaluation Framework**, an interactive platform designed to solve the problem of large-scale agent evaluation.  

Our framework provides developers and researchers with the tools to:  
- Automatically score agent responses  
- Compare different models head-to-head  
- Track performance over time  

Our solution directly addresses the hackathon challenge by accepting prompts, responses, and metadata to output an **interpretable performance report**.

---

## Key Features
Our framework is built as an interactive **Google Colab notebook** and includes:

- **Batch Processing Engine**: Processes thousands of responses from a CSV file, scoring each one across four key dimensions:  
  *Instruction-following, coherence, assumption control, and hallucination detection.*  

- **Live "Head-to-Head" Demo**: A user-friendly interface where a user can input a single prompt and have two different AI models generate responses in real-time. Both responses are instantly evaluated and displayed side-by-side.  

- **Performance Dashboard**: A visualization tab with a ranked leaderboard of all evaluated agents, providing a clear overview of which models perform best on average.  

- **Metadata Integration**: The evaluation process accepts metadata, allowing for more context-aware and nuanced judgments of an agent's response.  

---

## Technical Methodology
Our project was built with a focus on **innovation, speed, and technical depth**.

- **Core Technology**: Python, running in a Google Colab environment with a T4 GPU.  
- **Local LLMs with Ollama**: Runs powerful open-source models like *llama3:8b* and *phi3:mini* locally, ensuring zero cost and high flexibility.  
- **Asynchronous Processing for Speed**: In the Head-to-Head demo, all four LLM calls (two for generation, two for evaluation) are executed in parallel with Python's `asyncio`, reducing waiting time and ensuring a smooth user experience.  
- **Rich Interactive UI**: Built with `ipywidgets`, creating a responsive and intuitive application directly within the Colab notebook.  

---

## Setup and Usage Instructions
To run our project, follow these steps in a **Google Colab environment**:

1. **Open the File**  
   - Open the provided `Agentic_AI.py` file in Google Colab.  
   - Copy-paste the code inside a single cell.  

2. **Set the Runtime**  
   - Go to `Runtime -> Change runtime type`  
   - Select **T4 GPU**.  

3. **Run Cell 1 (Setup)**  
   - Installs Ollama, downloads the required AI models, and installs dependencies.  
   - Takes a few minutes.  

4. **Upload Data**  
   - Use the file browser (left panel) to upload `final_report_with_scores.csv`.  
   - Required for the Performance Dashboard.  

5. **Run Cell 2 (Application)**  
   - Launches the interactive, tabbed interface.  
   - You can now use the **Head-to-Head Demo** and **Performance Dashboard**.  

---

## Thank You
Thank you for evaluating our project!
