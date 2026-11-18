# UCF-Crime model

Both `config/config.yaml` and `config/ucf_ollama.yaml` use this fixed formula:

```text
action = 0.875 * visual_action + 0.125 * caption_action
final = 0.95 * action + 0.05 * calibrated_visual_normal_memory
```

The action scorer compares CLIP embeddings against 26 normal and 26 anomalous
phrases. It computes visual and caption evidence separately before combining
them. CLIP and Gemma weights remain frozen. Frames are sampled at 1 FPS, with
four-second windows and a one-second stride. Masking remains enabled as before.

The saved bank is `outputs/memory/ucf_visual_text_v1.npz`, built from 640 normal
UCF reference videos and calibrated on 160 other normal UCF videos. It stores
paired visual/text memory, but this configuration gives the text memory score
zero weight. Normal-memory calibration is part of the model and remains in use.
The bank is fixed during evaluation. Video checkpoints remain in
`outputs/checkpoints/ucf_ollama`.

The alert threshold is `0.5935198772337466`. It controls binary alerts and does
not affect ROC-AUC. The demo loads the same saved bank and scores the whole video.

The recorded result is **77.57% pooled native-frame ROC-AUC** across all 290 UCF
test videos (1,111,808 frames). The compact result, model settings, and memory
bank checksum are in `outputs/results/ucf_expanded95_summary.json`.

Provenance: prompts and fusion settings were selected during development using
some of these test videos. The all-video result is therefore a development
measurement, not an independent test estimate. This summary preserves the
existing measurement; cleanup did not run a new evaluation.

To run the configured UCF evaluation:

```bash
.venv/bin/python scripts/evaluate_ucf_multi_gpu.py --config config/ucf_ollama.yaml --gpus 0
```

See [the detached-screen guide](UCF_OLLAMA_SCREEN.md) for the server launcher
and [the multi-GPU guide](MULTI_GPU.md) for execution options. New UCF runs write
to the configured `outputs/ucf_expanded95_evaluation` directory.

XD-Violence evaluation uses its own normal reference bank. No XD job is started
by loading these settings.
