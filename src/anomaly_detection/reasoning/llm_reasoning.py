"""
Step 10: LLM Temporal Summarization / Deduction
------------------------------------------------------------------
Paper reference: "Encode 'Conceptual Logic' via LLM Temporal Summarization" -
a frozen LLM reviews a flagged event's description, the full 'Normal'
notebook, the similarity score, and surrounding context, then outputs a
natural-language explanation of the flag.
"""

import os
from typing import List
from anomaly_detection.utils.types import AnomalyResult


def explain_anomaly(result: AnomalyResult, notebook: List[str], context: str = "") -> AnomalyResult:
    """
    Builds a deduction prompt from result.event's description, the notebook,
    and result.anomaly_score, sends it to Gemini 2.5 Flash, and returns a copy of
    `result` with `.explanation` filled in.
    """
    if not result.is_anomaly:
        return result

    event_desc = result.event.description or "Uncaptioned event"
    notebook_str = "\n".join(f"- {rule}" for rule in notebook) if notebook else "- Default normal background"

    prompt = (
        "You are an expert AI Video Anomaly Rationalizer operating in a training-free VAD system.\n\n"
        f"Domain Constitution (Normal Activity Rules):\n{notebook_str}\n\n"
        f"Flagged Event Action: \"{event_desc}\"\n"
        f"Computed Anomaly Score: {result.anomaly_score:.3f}\n\n"
        "Instructions:\n"
        "Evaluate if the flagged event action violates the induced normal rules above.\n"
        "Output a concise 1-2 sentence human-friendly natural language explanation detailing why this event is anomalous."
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            result.explanation = response.text.strip()
            return result
        except Exception as e:
            result.explanation = (
                f"Action '{event_desc}' violates the established domain constitution "
                f"(notebook baseline). [LLM API call note: {e}]"
            )
            return result
    else:
        result.explanation = (
            f"Action '{event_desc}' violates the established domain constitution. "
            f"Set GEMINI_API_KEY environment variable for live Gemini LLM explanations."
        )
        return result

