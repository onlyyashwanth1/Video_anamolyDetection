# From One-Frame Event Captions to Temporal Window Analysis

Slide-by-slide content draft for a technical project review.

Scope: the event-based implementation at commit `a498216`, compared with the window-based implementation introduced in `2d4b031` and the current configuration. The optional multi-GPU additions in the working tree are covered separately. This is presentation content, not a rendered Project Kickoff template deck.

---

## Slide 1 — What changed in our video anomaly detector?

**Main message:** We now describe activity across time and combine normal-memory novelty with explicit action evidence.

- Before: one representative-frame caption for each detected event.
- Now: overlapping four-second windows with multiple frame captions and a temporal summary.
- Scoring: normal-memory comparison plus normal-versus-anomalous action similarity.
- Goal: capture more of the activity sequence and reduce reliance on scene familiarity.

**Speaker notes:** These are implemented architectural changes. Their effect on accuracy and runtime still needs a controlled benchmark. The pretrained models remain frozen.

**Suggested visual:** Two paths: “Event → single caption → memory score” and “Window → ordered captions → summary → memory + action score.”

---

## Slide 2 — What the old version actually did

**Main message:** The single-frame limitation was in captioning.

1. Sample video frames at approximately one frame per second.
2. Apply object-focused masking and multi-scale CLIP encoding.
3. Group frames into events using visual-similarity changes.
4. Send the middle sampled frame of each event to Gemma for a caption.
5. Compare the event representation and caption with normal textual memory.

**Speaker notes:** The old system already encoded multiple sampled frames and pooled their visual features. Saying “the entire detector used only one frame per event” would be inaccurate. One representative image supplied the event's natural-language description.

**Suggested visual:** Several frames feeding one pooled image embedding; only the middle frame feeding the caption.

---

## Slide 3 — Why change the event description?

**Main message:** A single image can omit the sequence that makes an activity meaningful.

- The chosen frame may occur before or after the critical action.
- Similar-looking scenes can contain different activities.
- Variable event boundaries depend on changes in visual similarity.
- Familiarity with the setting does not necessarily establish normal behavior.

**Speaker notes:** Illustrative scenario, not an observed model result: people approach, make physical contact, and separate. A caption of only the final frame might describe people standing nearby. Ordered observations can provide additional context, although they cannot recover motion that was never sampled.

**Suggested visual:** A three-step storyboard with the final frame highlighted as the old caption input.

---

## Slide 4 — The current pipeline

**Main message:** Local temporal context now feeds both description and scoring.

```text
Video → sample frames → object masking + CLIP features
                              ↓
                  4-second overlapping windows
                    ↙                       ↘
         pooled visual features       selected raw frames
                                            ↓
                                  cached frame captions
                                            ↓
                                  ordered temporal summary
                    ↘                       ↙
                  normal memory + action evidence
                              ↓
                   continuous window scores
                              ↓
                 average overlaps on native frames
```

**Speaker notes:** The summary receives timestamped caption text. This is not a new video motion backbone or joint multi-frame visual encoder. The captioning path uses raw frames; the visual feature path uses masking.

---

## Slide 5 — Change 1: predictable, overlapping windows

**Main message:** We replaced similarity-defined events with a regular temporal grid.

| Setting | Old version | Current default |
|---|---|---|
| Temporal unit | Similarity-defined event | Four-second window |
| Boundary rule | Coarse/fine similarity thresholds | Window starts every second |
| Caption input | One middle sampled frame | Up to four selected frames |
| End of video | Event segmentation boundaries | Explicit coverage through video end |

**Suggested visual:**

```text
Time:       0   1   2   3   4   5   6 seconds
Window A:   [---------------)
Window B:       [---------------)
Window C:           [---------------)
```

**Speaker notes:** At the configured sampling rate, a full window typically contains about four sampled frames. Short videos and the final window may contain fewer. Overlap offers shifted views of the same activity but increases the number of summaries.

---

## Slide 6 — Change 2: captions now preserve temporal order

**Main message:** We summarize a sequence of observations instead of describing one snapshot.

- Caption selected frames individually with Gemma.
- Attach timestamps and retain chronological order.
- Summarize the ordered captions into one short activity description.
- Reuse a frame's caption when it appears in overlapping windows.

**Speaker notes:** The prompt asks Gemma to use only supported actions, preserve uncertainty, and avoid inferring intent or unseen motion. It does not receive anomaly labels or the anomaly score. A one-frame window reuses that caption directly. The serial caption cache is in memory for one processing call; it does not provide restart recovery.

**Suggested visual:** Four timestamped caption cards feeding one summary card. Any example captions should be labeled illustrative.

---

## Slide 7 — Change 3: action evidence supplements memory

**Main message:** The detector now considers both familiarity and the type of activity.

| Component | Question it approximates | Evidence |
|---|---|---|
| Memory score | How different is this from normal descriptions? | Visual and caption embeddings compared with normal memory |
| Action score | Is this closer to anomalous or normal action prompts? | Visual and caption embeddings compared with both prompt groups |

**Default combination:**

```text
Combined score = 0.5 × memory score + 0.5 × action score
Alert when score > 0.50
```

**Speaker notes:** Examples of normal prompts include walking and peaceful conversation; anomalous prompts include punching, collisions, fire, and property damage. For each probe, the action scorer subtracts the best normal similarity from the best anomalous similarity, applies a sigmoid with temperature 0.1, and averages the two probe scores. These values are heuristic rankings, not probabilities. Weights and thresholds require validation.

---

## Slide 8 — Change 4: memory and frame-level evaluation

**Main message:** Memory construction and evaluation must match the new temporal representation.

- Rebuild normal memory using window summaries.
- Use a separate bank: `ucf_crime_windows_v2.npz`.
- Average overlapping continuous window scores at each native video frame.
- Retain individual score components and raw action margins for inspection.

**Speaker notes:** Old banks are preserved, and incompatible artifacts are rejected. Benchmark inference does not update normal memory. In the demo, windows crossing the normal-observation boundary are excluded, and automatic promotion of repeated anomaly alerts is disabled by default. Optional LLM verdict review is disabled in the default benchmark; Gemma captioning remains enabled.

**Suggested visual:** Three overlapping window scores feeding their mean at a shared frame.

---

## Slide 9 — Benefits we expect and costs we accept

**Main message:** We added temporal evidence, with additional computation and remaining blind spots.

| Change | Intended benefit | Limitation or cost |
|---|---|---|
| Multiple captioned frames | More observations of the activity | One FPS can miss fast actions |
| Ordered temporal summary | More context about progression | Errors can propagate from captions |
| Overlapping windows | Less dependence on a single boundary | More summaries and delayed completed-window output |
| Action prompts | Evidence beyond scene familiarity | Coverage is limited to the prompt vocabulary |
| Caption/feature reuse | Avoid repeated work within a video | Full sampled video still occupies host memory |

**Speaker notes:** Accuracy improvement is a hypothesis to test. There is no added audio input or learned motion backbone. Mean pooling of visual embeddings itself does not encode temporal order; the ordered caption summary supplies the explicit sequence description.

---

## Slide 10 — What the current run establishes

**Main message:** The selected log shows memory-building progress, not a completed new-version benchmark.

- Input validation recorded 800 normal-memory videos and 290 test videos.
- At inspection, the latest progress marker was `Memory [323/800]`.
- That marker is written before processing the video; it does not certify 323 completed videos.
- No final window-version evaluation metrics were found in the selected log.

**Speaker notes:** Source: `outputs/logs/ucf_windows_v2_20261017_102706.log`, inspected on 20 October 2026. This is frozen-model processing and memory construction, not model training. Historical evaluation outputs exist, but they do not demonstrate a before/after gain for this new run. Progress is a snapshot and may change.

**Suggested visual:** “Input validation → memory construction → test evaluation → comparison,” with memory construction highlighted.

---

## Slide 11 — How we will measure the contribution of each change

**Main message:** Compare variants on the same split to separate the effects.

| Variant | Temporal unit | Description | Scoring |
|---|---|---|---|
| Old architecture | Events | Middle frame | Memory |
| Windows only | Sliding windows | Middle frame | Memory |
| Temporal captions | Sliding windows | Ordered caption summary | Memory |
| Action only | Sliding windows | Ordered caption summary | Action |
| Current combined | Sliding windows | Ordered caption summary | Memory + action |

- Report pooled frame-level ROC-AUC and average precision.
- Report precision, recall, and F1 at a validation-selected threshold.
- Measure processing time and resource use.
- Inspect missed short actions and false alarms in familiar scenes.

**Speaker notes:** Rebuild compatible memory for each temporal/caption configuration. Select prompts, weights, temperature, and threshold using a separate development split. The available old-architecture mode is useful for controlled comparisons, but it is not a guarantee of bit-identical historical behavior after other implementation fixes.

---

## Slide 12 — Additional engineering: optional multi-GPU execution

**Main message:** The working tree also contains an optional route to parallelize costly video processing.

- UCF-Crime and XD-Violence entry points distribute whole videos across GPUs.
- Each worker keeps its own model replica; longer videos are queued first.
- Completed-video checkpoints allow later runs to reuse expensive processing.
- Final normal-bank assembly and scoring remain serial.

**Speaker notes:** These additions are currently untracked in Git and separate from the committed window architecture. They do not change sampling, caption prompts, window geometry, or scoring settings. Each GPU must fit a full model replica. The selected serial run cannot join the new queue or export its unfinished in-memory work. Real-model speedup has not been established.

---

## Presenter reference — evidence map

These references support the content above; this section need not be projected.

| Slides | Source |
|---|---|
| 1–3 | `git show a498216:src/anomaly_detection/processing.py`; `git show a498216:src/anomaly_detection/familiarisation/domain_familiarisation.py` |
| 4–5 | [Current processor](../src/anomaly_detection/processing.py), [window implementation](../src/anomaly_detection/segmentation/sliding_windows.py), [configuration](../config/config.yaml) |
| 6 | [Caption selection, caching, and summary prompt](../src/anomaly_detection/familiarisation/domain_familiarisation.py) |
| 7 | [Action scorer](../src/anomaly_detection/inference/action_scoring.py), [score fusion](../src/anomaly_detection/inference/adaptive_inference.py) |
| 8–9, 11 | [Window pipeline documentation](WINDOW_PIPELINE.md), [evaluation runner](../src/anomaly_detection/evaluation/runner.py) |
| 10 | [Selected run log](../outputs/logs/ucf_windows_v2_20261017_102706.log), [memory builder](../src/anomaly_detection/evaluation/bank.py) |
| 12 | [Multi-GPU documentation](MULTI_GPU.md), [scheduler](../src/anomaly_detection/evaluation/multi_gpu.py), [video cache](../src/anomaly_detection/evaluation/video_cache.py) |

Editorial note: older sections of `PROJECT_STATUS.md` describe superseded captioning backends and historical completion claims. This draft prioritizes executable code, current configuration, the window-specific documentation, and the selected log.
