"""
Step 10: LLM Temporal Summarization / Deduction
------------------------------------------------------------------
Paper reference: "Encode 'Conceptual Logic' via LLM Temporal Summarization" -
a frozen Multimodal/LLM reviews a flagged event's description, the full
'Normal' notebook, the similarity score, and surrounding context, then
outputs a natural-language explanation of the flag.

Implementation:
    Uses local Gemma 4 31B loaded directly via transformers
    (see local_gemma.py). No external server required.
"""

from typing import List

from anomaly_detection.utils.types import AnomalyResult


# ------------------------------------------------------------------
# Step 10: Temporal Summarization / Deduction
# ------------------------------------------------------------------

def explain_anomaly(
    result: AnomalyResult,
    notebook: List[str],
    context: str = "",
    strict: bool = False,
) -> AnomalyResult:
    """
    Builds a deduction prompt from:

        - flagged event description
        - Domain Constitution / Normal notebook
        - anomaly score
        - surrounding temporal context

    Sends the prompt to the local Gemma 4 31B model and returns the
    result with `.explanation` populated.

    Parameters
    ----------
    result:
        AnomalyResult containing the flagged event and anomaly score.

    notebook:
        List of normal activity descriptions/rules.

    context:
        Optional surrounding temporal context.

    Returns
    -------
    AnomalyResult
        The original result with `.explanation` filled in.
    """

    # Nothing to explain if the event wasn't flagged.
    if not result.is_anomaly:
        return result

    event_desc = (
        result.event.description
        if result.event.description
        else "Uncaptioned event"
    )

    notebook_str = (
        "\n".join(f"- {rule}" for rule in notebook)
        if notebook
        else "- Default normal background"
    )

    context_str = (
        context.strip()
        if context and context.strip()
        else "No additional temporal context available."
    )

    # --------------------------------------------------------------
    # Deduction prompt
    # --------------------------------------------------------------

    prompt = (
        "You are an expert AI Video Anomaly Rationalizer operating "
        "in a training-free video anomaly detection system.\n\n"

        "Your task is to explain why a flagged event may be anomalous "
        "relative to the learned normal behavior of the environment.\n\n"

        f"Domain Constitution (Normal Activity Rules):\n"
        f"{notebook_str}\n\n"

        f"Flagged Event Action:\n"
        f"\"{event_desc}\"\n\n"

        f"Computed Anomaly Score:\n"
        f"{result.anomaly_score:.3f}\n\n"

        f"Surrounding Temporal Context:\n"
        f"{context_str}\n\n"

        "Instructions:\n"
        "1. Compare the flagged event against the normal activity rules.\n"
        "2. Use the anomaly score only as supporting evidence; do not treat it as proof by itself.\n"
        "3. Consider the surrounding temporal context when provided.\n"
        "4. Decide if the event is a true anomaly (violates the normal rules) or normal (fits/aligns with one or more normal rules).\n"
        "5. Output your response in the following exact format:\n"
        "VERDICT: [ANOMALY or NORMAL]\n"
        "EXPLANATION: [A concise 1-2 sentence explanation of your decision. If VERDICT is NORMAL, explain that the evidence is insufficient to justify an anomaly or that the event aligns with normal behavior.]\n"
    )

    # --------------------------------------------------------------
    # Gemma 4 31B via local transformers
    # --------------------------------------------------------------

    try:
        from anomaly_detection.reasoning import local_gemma
        response = local_gemma.generate_text(prompt).strip()
        
        verdict = "ANOMALY"
        explanation = ""

        if "VERDICT:" in response:
            lines = response.split("\n")
            for line in lines:
                if line.startswith("VERDICT:"):
                    verdict_str = line.replace("VERDICT:", "").strip().upper()
                    if "NORMAL" in verdict_str:
                        verdict = "NORMAL"
                    elif "ANOMALY" in verdict_str:
                        verdict = "ANOMALY"
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()
            
            if not explanation and "EXPLANATION:" in response:
                parts = response.split("EXPLANATION:")
                if len(parts) > 1:
                    explanation = parts[1].replace("\n", " ").strip()

        # Fallback if format is not strictly followed
        if not explanation:
            explanation = response.replace("\n", " ").strip()
            if any(phrase in explanation.lower() for phrase in ["insufficient evidence", "no evidence", "unsupported", "normal behavior", "aligns with"]):
                verdict = "NORMAL"

        if verdict == "NORMAL":
            result.is_anomaly = False

        if explanation:
            result.explanation = explanation
        else:
            result.explanation = (
                f"Event '{event_desc}' was flagged as anomalous "
                "relative to the learned normal activity."
            )

        return result

    except Exception as e:
        if strict:
            raise RuntimeError("Gemma anomaly review failed") from e
        result.explanation = (
            f"Action '{event_desc}' was flagged as anomalous relative "
            f"to the established domain constitution "
            f"(anomaly score: {result.anomaly_score:.3f}). "
            f"[Local Gemma Note: {e}]"
        )
        return result
