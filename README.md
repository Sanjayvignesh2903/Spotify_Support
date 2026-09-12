# Spotify_Support
AI-powered customer support triage dashboard for the Hiver SDE Intern assessment.

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

Escalate Path (Hardcoded Fallback): If the routing engine detects a sensitive issue, the generation phase is bypassed. The system outputs a strict, hardcoded template directing the user to a secure DM or the Spotify Smarthelp portal.

2.4. LLM-as-a-Judge Evaluation
An integrated evaluation endpoint allows operators to grade the drafted response on a 1-to-5 scale based on brand tone, helpfulness, and safety.
