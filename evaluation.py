# evaluation.py - Hiver Baseline & LLM Metrics Evaluation
import time
import json
import pandas as pd
from groq import Groq
from sklearn.metrics import classification_report

# Initialize Groq Client (Hardcoded for local testing as per app.py)
client = Groq(api_key="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")

def classify_ticket(customer_message: str, model: str = "openai/gpt-oss-20b") -> dict:
    """Zero-shot LLM classifier for Spotify support tickets."""
    prompt = f"""You are an automated support classifier for Spotify.
Analyze this tweet and classify the intent and action.

Taxonomy intents:
- account_access
- billing_subscription
- playback_streaming_issue
- playlist_library_management
- content_availability
- general_feedback_ui

Actions:
- Escalate: ONLY for sensitive account security, billing disputes, or account lookup.
- Auto: for troubleshooting, links, feedback, or general availability.

Tweet: "{customer_message}"

Output JSON strictly:
{{"intent": "<intent>", "action": "<Auto|Escalate>", "reason": "<brief justification>"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def run_evaluation():
    # Load the labeled evaluation dataset
    file_path = "spotify_labeled_tweets.csv"
    try:
        df_eval = pd.read_csv(file_path).dropna(subset=['Customer_Message', 'True_Intent'])
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Ensure it is in the same folder.")
        return

    predictions = []
    actions = []
    print(f"Running LLM evaluation on {len(df_eval)} tickets...")

    for idx, row in df_eval.iterrows():
        try:
            res = classify_ticket(row['Customer_Message'])
            pred_intent = res.get('intent', 'general_feedback_ui')
            pred_action = res.get('action', 'Auto')
        except Exception as e:
            pred_intent = 'general_feedback_ui'
            pred_action = 'Auto'
        
        predictions.append(pred_intent)
        actions.append(pred_action)
        
        # Rate limit safety buffer for API
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(df_eval)} tickets...")
            time.sleep(1)

    # Attach predictions to the dataframe
    df_eval['LLM_Predicted_Intent'] = predictions
    df_eval['LLM_Predicted_Action'] = actions

    # Print Final Metrics
    print("\n" + "=" * 50)
    print("--- LLM AGENT INTENT CLASSIFICATION METRICS ---")
    print("=" * 50)
    print(classification_report(df_eval['True_Intent'], df_eval['LLM_Predicted_Intent'], zero_division=0))

    if 'Action' in df_eval.columns:
        print("\n" + "=" * 50)
        print("--- LLM AGENT ACTION (ROUTING) METRICS ---")
        print("=" * 50)
        print(classification_report(df_eval['Action'], df_eval['LLM_Predicted_Action'], zero_division=0))

    # Save output for failure analysis reporting
    output_path = "spotify_eval_results_with_predictions.csv"
    df_eval.to_csv(output_path, index=False)
    print(f"\nSaved evaluation predictions for manual review to: {output_path}")

if __name__ == "__main__":
    run_evaluation()