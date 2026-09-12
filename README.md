# 🎧 Spotify AI Support Operations Hub

An AI-powered customer support triage dashboard built for the Hiver SDE Intern assessment.

This repository contains a full-stack pipeline featuring a zero-shot LLM intent classifier, a TF-IDF Retrieval-Augmented Generation (RAG) module, and an interactive Streamlit UI mimicking a production operations hub.

## 📁 Project Structure

* **`app.py`**: The interactive Streamlit dashboard for real-time ticket triage and LLM-as-a-judge evaluation.
* **`evaluation.py`**: The script used to generate the baseline and LLM metrics on the golden set.
* **`spotify_labeled_tweets.csv`**: The 200-row manually annotated golden dataset.
* **`spotify_knowledge_base.csv`**: The 42k-row historical context dataset used for TF-IDF RAG retrieval.
* **`Spotify_Agent_Report.pdf`**: The comprehensive 6-page architectural and evaluation report.

## 🚀 Installation & Setup

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install scikit-learn pandas groq streamlit

3. The Groq API key is temporarily hardcoded in the script for seamless evaluation by the review team.

## 💻 How to Run

To launch the interactive triage UI, run the following command in your terminal:

```bash
streamlit run app.py


Once launched, use the Quick Ticket Scenarios buttons in the dashboard to test auto-resolution and the strict security escalation protocols.

📊 Evaluation Summary
Trivial Baseline: 41.0% Accuracy

Keyword Baseline: 56.0% Accuracy

Zero-Shot LLM Agent: 64.0% Accuracy (0.66 Macro F1)

Routing (Action) Accuracy: 79.0%
