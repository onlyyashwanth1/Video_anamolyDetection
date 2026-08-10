"""
Training-free, domain-independent video anomaly detection framework.

Package layout mirrors the paper's pipeline, one module per stage:
    ingestion         -> Step 1  (Live Input Stream)
    encoding          -> Step 2  (Frozen Foundational Encoder / CLIP)
    masking           -> Step 4  (Spatial Object-Centric Masking)
    segmentation      -> Step 3  (Automated Temporal Decomposition / GEBD+HGTree)
    familiarisation   -> Steps 5-6 (Domain Familiarisation / Domain Constitution)
    memory            -> Step 7  (Dynamic Textual Memory Bank + familiarity counter)
    inference         -> Steps 8-9 (Contrastive Probing + Soft Threshold Gating)
    reasoning         -> Step 10 (LLM Temporal Summarization / Deduction)

See docs/PROJECT_STATUS.md for what's implemented vs. still a stub.
"""
