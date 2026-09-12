# Spotify_Support
AI-powered customer support triage dashboard for the Hiver SDE Intern assessment.

**#1. Executive Summary**
This report details the design, engineering, and evaluation of an AI-powered customer support triage system for Spotify's Twitter channel (@SpotifyCares). The objective of this system is to reduce human agent workload by automatically resolving tier-1 troubleshooting queries while securely escalating sensitive account and billing issues to human operators.

The delivered solution is a full-stack pipeline featuring a zero-shot Large Language Model (LLM) intent classifier, a TF-IDF Retrieval-Augmented Generation (RAG) module for historical context, and an automated LLM-as-a-judge Quality Assurance (QA) evaluator. The backend is wrapped in a highly interactive, consumer-facing Streamlit dashboard mimicking the Spotify Support UI. Empirical evaluation on a 200-row golden dataset demonstrates that the LLM agent achieved a 64% intent classification accuracy and a 79% routing accuracy, significantly outperforming heuristic baselines while adhering to strict security guardrails.
