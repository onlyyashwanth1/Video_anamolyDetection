"""
Step 10: LLM Temporal Summarization / Deduction
------------------------------------------------------------------
Paper reference: "Encode 'Conceptual Logic' via LLM Temporal Summarization" -
a frozen Multimodal/LLM reviews a flagged event's description, the full
'Normal' notebook, the similarity score, and surrounding context, then
outputs a natural-language explanation of the flag.

Implementation:
    Uses local Gemma 4 31B through Ollama.
"""

from typing import List

from anomaly_detection.utils.types import AnomalyResult


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OLLAMA_LLM_MODEL = "gemma4:31b-it-q8_0"


# ------------------------------------------------------------------
# Step 10: Temporal Summarization / Deduction
# ------------------------------------------------------------------

def explain_anomaly(
    result: AnomalyResult,
    notebook: List[str],
    context: str = "",
) -> AnomalyResult:
    """
    Builds a deduction prompt from:

        - flagged event description
        - Domain Constitution / Normal notebook
        - anomaly score
        - surrounding temporal context

    Sends the prompt to Gemma 4 31B through Ollama and returns the
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
        "2. Use the anomaly score only as supporting evidence; do not "
        "treat it as proof by itself.\n"
        "3. Consider the surrounding temporal context when provided.\n"
        "4. Explain the specific mismatch between the event and normal "
        "behavior.\n"
        "5. If the event does not clearly violate the normal rules, say "
        "that the evidence is insufficient rather than inventing a reason.\n"
        "6. Output only a concise 1-2 sentence explanation.\n"
    )

    # --------------------------------------------------------------
    # Gemma 4 31B via Ollama
    # --------------------------------------------------------------

    try:
        import ollama

        response = ollama.generate(
            model=OLLAMA_LLM_MODEL,
            prompt=prompt,
        )

        explanation = (
            response.get("response", "")
            .strip()
            .replace("\n", " ")
        )

        if explanation:
            result.explanation = explanation
        else:
            result.explanation = (
                f"Event '{event_desc}' was flagged as anomalous "
                "relative to the learned normal activity."
            )

        return result

    except Exception as e:
        result.explanation = (
            f"Action '{event_desc}' was flagged as anomalous relative "
            f"to the established domain constitution "
            f"(anomaly score: {result.anomaly_score:.3f}). "
            f"[Ollama LLM Note: {e}]"
        )

        return result