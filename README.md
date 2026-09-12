# Spotify_Support
AI-powered customer support triage dashboard for the Hiver SDE Intern assessment.

**1. Executive Summary**

This report details the design, engineering, and evaluation of an AI-powered customer support triage system for Spotify's Twitter channel (@SpotifyCares). The objective of this system is to reduce human agent workload by automatically resolving tier-1 troubleshooting queries while securely escalating sensitive account and billing issues to human operators.

The delivered solution is a full-stack pipeline featuring a zero-shot Large Language Model (LLM) intent classifier, a TF-IDF Retrieval-Augmented Generation (RAG) module for historical context, and an automated LLM-as-a-judge Quality Assurance (QA) evaluator. The backend is wrapped in a highly interactive, consumer-facing Streamlit dashboard mimicking the Spotify Support UI. Empirical evaluation on a 200-row golden dataset demonstrates that the LLM agent achieved a 64% intent classification accuracy and a 79% routing accuracy, significantly outperforming heuristic baselines while adhering to strict security guardrails.

**2. System Architecture & Engineering**
The system operates on a four-stage pipeline designed for low latency, high relevance, and strict policy compliance.

**2.1. Ingestion & Zero-Shot Triage (Routing)**
Incoming customer tweets are passed to a lightweight, fast LLM operating as a zero-shot classifier. The prompt forces a strict JSON output.

Intent Classification: Maps the tweet to one of six taxonomy classes (e.g., account_access, billing_subscription, playback_streaming_issue).

Action Routing: Categorizes the ticket as either Auto (safe to resolve via AI) or Escalate (requires human intervention).

**2.2. Context Retrieval (TF-IDF RAG)**
Instead of relying on LLM parametric memory, the system uses a RAG architecture querying a ~42,800-row historical dataset of verified Spotify support resolutions.

The text is pre-processed (handles removed, URLs stripped, normalized).

A TfidfVectorizer (configured with max_features=15000, ngram_range=(1, 2), and sublinear_tf=True) transforms the corpus.

Cosine similarity retrieves the top-2 most relevant historical interactions to ground the AI's response in verified brand policies.

**2.3. Guardrailed Response Generation**
Auto Path: For standard issues, the LLM drafts a concise (<280 character) tweet utilizing the retrieved TF-IDF context, adhering to a friendly, proactive brand tone.

Escalate Path (Hardcoded Fallback): If the routing engine detects a sensitive issue, the generation phase is bypassed. The system outputs a strict, hardcoded template directing the user to a secure DM or the Spotify Smarthelp portal.

2.4. LLM-as-a-Judge Evaluation
An integrated evaluation endpoint allows operators to grade the drafted response on a 1-to-5 scale based on brand tone, helpfulness, and safety.
