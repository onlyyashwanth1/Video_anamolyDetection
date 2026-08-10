"""
Step 10: LLM Temporal Summarization / Deduction  [NOT YET IMPLEMENTED]
------------------------------------------------------------------
Paper reference: "Encode 'Conceptual Logic' via LLM Temporal
Summarization" - a frozen LLM reviews a flagged event's description, the
full 'Normal' notebook, the similarity score, and surrounding context
(from the event tree), then outputs a refined anomaly score (0.0-1.0)
plus a natural-language explanation of the flag.

Where this fits in the pipeline:
    inference (Steps 8-9) flags an event -> reasoning (THIS MODULE) is
    called ONLY on flagged events, to produce the final human-readable
    explanation. Normal (non-flagged) events skip this module entirely.

Planned interface:
    def explain_anomaly(result: AnomalyResult, notebook: list[str],
                         context: str = "") -> AnomalyResult
        Builds a prompt from result.event's description, the notebook,
        result.anomaly_score, and any surrounding context, sends it to an
        LLM, and returns a copy of `result` with `.explanation` filled in
        (and optionally an LLM-refined `.anomaly_score`).

TODO:
    - Pick and integrate an LLM API (e.g. Claude/GPT via API) for
      the reasoning call
    - Design the exact prompt template (see the framework doc's example
      prompt for a starting point)
    - Decide whether to let the LLM override the numeric score from
      Step 9, or only ever add the explanation text
"""

from typing import List

from anomaly_detection.utils.types import AnomalyResult


def explain_anomaly(result: AnomalyResult, notebook: List[str], context: str = "") -> AnomalyResult:
    raise NotImplementedError(
        "LLM reasoning/explanation is not implemented yet. "
        "See this module's docstring for the planned interface and TODOs."
    )
