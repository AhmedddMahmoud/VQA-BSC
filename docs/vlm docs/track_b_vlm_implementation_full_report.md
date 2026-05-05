# Track B (Pretrained VLM) — Full Implementation Report

## 0) Purpose of this document

This file is an exhaustive handover/report of exactly what was implemented for **Track B** in this repository, why each change was made, how each new file works, what was intentionally not changed, how the evaluation path behaves, what was validated, and what is still missing/blocking.

This report is intentionally detailed and should be considered the source-of-truth for the first Track B iteration.

---

## 1) Context and constraints respected

The implementation was done under the following strict constraints:

1. **Do not restart from zero.**
2. **Do not rewrite Track A.**
3. **Do not merge VLM into classical GRU+attention model.**
4. Build a **clean parallel Track B path**.
5. Start with **inference-only** VLM evaluation (no fine-tuning).
6. Reuse existing evaluation/scoring utilities where valid.
7. Save artifacts in existing repo style:
   - predictions JSONL
   - qualitative failures JSONL
   - metrics summary JSON

All these constraints were followed.

---

## 2) What existed before Track B changes

The repository already had:

- Classical Track A model and training:
  - `src/models/multihop_vqa.py`
  - `src/train/train_vqa.py`
  - `src/train/eval_vqa.py`
- Top-K classifier dataset path:
  - `src/datasets/vqa_dataset.py`
- Reusable utilities:
  - `src/utils/metrics.py`
  - `src/utils/vocab.py`
- Config-driven setup under `configs/`.
- Standard output folders under `outputs/`.

### Important existing design assumption discovered

`src/datasets/vqa_dataset.py` is classifier-centric and includes top-K filtering:

- It requires `answer_to_idx`.
- It drops samples where majority answer is not in top-K.
- It returns a `label` (`answer_idx`) tensor for classification loss.

This is incompatible with generative VLM evaluation where we want:
- image + question,
- generated free-form answer,
- scoring against full gold-answer list,
- **no top-K label dependency**.

This is why a **parallel Track B dataset** was created instead of altering Track A dataset logic.

---

## 3) Files created/updated (complete list)

### New files created

1. `src/datasets/vqa_vlm_eval_dataset.py`
2. `src/models/vlm/__init__.py`
3. `src/models/vlm/paligemma_runner.py`
4. `src/train/eval_vlm_vqa.py`
5. `configs/track_b_paligemma_vqav2_eval.json`
6. `vlm docs/track_b_vlm_implementation_full_report.md` (this document)

### Existing file updated

7. `requirements.txt`
   - added `sentencepiece>=0.2.0`

### Files intentionally untouched (examples, not exhaustive)

- `src/datasets/vqa_dataset.py`
- `src/train/train_vqa.py`
- `src/train/eval_vqa.py`
- `src/models/multihop_vqa.py`
- Classical Track A config files under `configs/version1_*.json`

This preserves scientific separation: Track A control remains intact.

---

## 4) Detailed file-by-file explanation

## 4.1 `src/datasets/vqa_vlm_eval_dataset.py`

### Why this file was created

To provide a **Track B-only evaluation dataset** that:

- does **not** depend on top-K answer vocab,
- does **not** produce classifier labels,
- loads image + question + full gold answers,
- preserves metadata needed for evaluation logs.

### Main class: `VQAv2VLMEvalDataset`

Constructor args:
- `questions_path`
- `annotations_path`
- `images_dir`
- `max_samples`

### What it loads

- Reads VQAv2 questions JSON and annotations JSON.
- Maps questions by `question_id`.
- Iterates annotations and joins with question text.

### Per-sample fields stored

Each sample includes:
- `question_id`
- `question_text`
- `image_id`
- `all_gold_answers` (normalized)
- `gold_majority_answer` (via existing `extract_majority_answer`)
- `answer_type` (from annotation if present)

### Image resolution behavior

`_resolve_image_path(image_id)` tries in order:
1. `COCO_train2014_{id}.jpg`
2. `COCO_val2014_{id}.jpg`
3. fallback train-style path string

This mirrors existing mixed-folder tolerance used in Track A evaluation.

### `__getitem__`

Returns:
- image as PIL RGB object (`image`)
- `image_path`, `image_id`
- question fields
- answer fields (`all_gold_answers`, `gold_majority_answer`, `answer_type`)

### Collate function

`vqa_vlm_eval_collate_fn(batch)` returns lists:
- `question_ids`, `question_texts`
- `image_ids`, `image_paths`, `images`
- `all_gold_answers`, `gold_majority_answers`, `answer_types`

### Why this is correct for Track B

- Keeps generative evaluation independent from classifier assumptions.
- Avoids mutating classical data pipeline.
- Keeps output metadata rich for analysis and failure logging.

---

## 4.2 `src/models/vlm/__init__.py`

### Why this file was created

To define a clean namespace for VLM models and export `PaliGemmaRunner`.

### Content

- Imports `PaliGemmaRunner` from `paligemma_runner.py`
- Sets `__all__ = ["PaliGemmaRunner"]`

### Why it matters

- Keeps `src/models/vlm` modular.
- Allows `from src.models.vlm import PaliGemmaRunner`.

---

## 4.3 `src/models/vlm/paligemma_runner.py`

### Why this file was created

This is the dedicated Track B runtime wrapper that:

- loads pretrained PaliGemma,
- processes image+text inputs,
- runs deterministic generation,
- returns answer strings.

No coupling to Track A architecture.

### Dataclass: `GenerationConfig`

Fields:
- `max_new_tokens` (default `8`)
- `do_sample` (default `False`)
- `num_beams` (default `1`)

These defaults enforce deterministic baseline-like inference behavior.

### Class: `PaliGemmaRunner`

Constructor args:
- `model_name_or_path`
- `device` (`torch.device`)
- `generation` (`GenerationConfig`)
- `prompt_template` (default `"{question}"`)

### Constructor behavior

1. Checks whether `PaliGemmaForConditionalGeneration` is available in installed transformers build.
2. Loads `AutoProcessor.from_pretrained(model_name_or_path)`.
3. Loads model with dtype:
   - `float16` on CUDA
   - `float32` on CPU
4. Moves model to device and sets `eval()` mode.

### Error handling added

If model/processor loading fails (for example gated repo access), code raises:

- `RuntimeError` with actionable guidance:
  - run `huggingface-cli login`
  - ensure HF account has access to target model repo

This was added after observed runtime failure with 401/gated access.

### Prompt handling

- `_make_prompt(question)` applies `prompt_template.format(question=question)`.

### Generation method

`generate_answers(images, questions)`:

1. Validates equal batch lengths.
2. Builds prompts from questions.
3. Calls processor with images + text, padded tensors.
4. Sends tensors to device.
5. Calls `model.generate(...)` with deterministic settings.
6. Computes prompt token lengths from `input_ids`/pad token.
7. Decodes only completion slice (not full prompt echo).
8. Returns stripped answer strings.

### Why this implementation is minimal and appropriate

- No fine-tuning logic.
- No retrieval/OCR/KB integration.
- No classifier output projection.
- Single responsibility: inference generation for Track B.

---

## 4.4 `src/train/eval_vlm_vqa.py`

### Why this file was created

To provide a dedicated Track B evaluation entrypoint that mirrors output behavior of classical evaluation while using generative VLM predictions.

### Dataclass: `TrackBVLMEvalConfig`

Fields include:
- runtime: `seed`, `device`
- model: `model_name_or_path`, `prompt_template`
- generation: `generation_max_new_tokens`, `generation_do_sample`, `generation_num_beams`
- data: val paths + subset size
- dataloader: `batch_size`, `num_workers`
- outputs: `logs_dir`, `predictions_dir`, `qualitative_failures_to_save`
- feature toggles: `save_outputs`, `compute_supplementary_text_overlap`

`from_json()` loads this config from file.

### CLI arguments

- `--config` (default `configs/track_b_paligemma_vqav2_eval.json`)
- `--subset-val` override
- `--split-name` (default `val`)

### Helper functions

- `resolve_device()` -> `cuda` if available else `cpu`
- `set_seed()` -> Python/NumPy/Torch seeding

### Core function: `evaluate_vlm_on_loader(...)`

Per sample:
1. Generate answer via `PaliGemmaRunner`.
2. Keep both:
   - `predicted_answer_raw`
   - normalized `predicted_answer`
3. Normalize gold majority + all gold answers.
4. Compute:
   - per-sample exact normalized match vs majority (`exact_match_correct`)
   - per-sample VQA soft score (`vqa_soft_score`)
5. Build prediction rows and failure rows.

After loop computes:
- `top1_accuracy` (implemented as normalized exact-match rate vs gold majority)
- `vqa_soft_accuracy`
- `answer_type_breakdown` (reused utility)
- optional supplementary overlap metrics (BLEU/ROUGE-L/METEOR)

### Artifact saving

If `save_outputs`:

- predictions JSONL:
  - `outputs/predictions/eval_track_b_paligemma_predictions_<split>_<timestamp>.jsonl`
- failures JSONL (limited by `qualitative_failures_to_save`):
  - `outputs/predictions/eval_track_b_paligemma_failures_<split>_<timestamp>.jsonl`
- metrics summary JSON:
  - `outputs/logs/eval_track_b_paligemma_metrics_<split>_<timestamp>.json`

Same artifact style as existing Track A eval.

### Console output

Prints:
- sample count
- top1 accuracy
- VQA soft accuracy
- answer-type table
- supplementary metrics (if enabled)
- saved file paths

### Why this file is needed

- Keeps Track B eval isolated from classical model loading.
- Reuses validated metrics utilities.
- Preserves reporting conventions.

---

## 4.5 `configs/track_b_paligemma_vqav2_eval.json`

### Why this file was created

Track B needs its own config file rather than overloading classical configs.

### Key values

- `model_name_or_path`: `google/paligemma-3b-ft-vqav2-224`
- deterministic generation defaults:
  - `generation_do_sample: false`
  - `generation_num_beams: 1`
  - `generation_max_new_tokens: 8`
- `batch_size: 1` (safe default for large VLM)
- val subset default `64`
- dataset paths aligned with current repo layout under `datasets/train2014/...`
- output dirs align with existing `outputs/logs` and `outputs/predictions`

### Why this matters

- Keeps evaluation reproducible and explicit.
- Cleanly separates Track B parameters from Track A.

---

## 4.6 `requirements.txt` (updated)

### Change

Added:
- `sentencepiece>=0.2.0`

### Why

Many HF processors/tokenizers require SentencePiece backend; this avoids tokenizer runtime issues when loading VLM checkpoints.

---

## 5) Reused existing utilities (unchanged)

The following existing functions are reused as-is:

From `src/utils/metrics.py`:
- `normalize_answer_for_eval`
- `exact_match_after_normalization`
- `compute_vqa_soft_accuracy`
- `compute_answer_type_breakdown`
- `compute_supplementary_text_overlap_metrics`
- `save_jsonl`

From `src/utils/vocab.py`:
- `normalize_answer`
- `extract_majority_answer`

This reuse avoided duplication and kept metric behavior consistent with existing project logic.

---

## 6) Validation performed

## 6.1 Static/syntax and import-level checks

Executed successfully in project venv:

- `./.venv/bin/python -m src.train.eval_vlm_vqa --help`

The command printed expected CLI options, confirming import path is valid.

## 6.2 Runtime sanity attempt

Executed:

- `./.venv/bin/python -m src.train.eval_vlm_vqa --config configs/track_b_paligemma_vqav2_eval.json --subset-val 1`

Observed failure:
- `401 Unauthorized / GatedRepoError` for model `google/paligemma-3b-ft-vqav2-224`

Interpretation:
- Pipeline reaches model loading stage correctly.
- Blocker is HF model access/auth, not code wiring.

### Improvement made from this failure

`PaliGemmaRunner` now wraps model load errors with explicit action message about HF login/access.

---

## 7) What is complete vs what is missing

## 7.1 Complete

1. Track B has a clean standalone evaluation codepath.
2. Track A remains untouched.
3. Generative dataset path without top-K label assumptions exists.
4. Deterministic generation settings are in place.
5. Existing scoring infrastructure is reused.
6. Artifact saving conventions are preserved.
7. Config-driven execution is implemented.

## 7.2 Missing / blocked / next required steps

### A) External access blocker

- Need Hugging Face authentication and approved access to `google/paligemma-3b-ft-vqav2-224`.

Without this, full end-to-end evaluation cannot run.

### B) Not implemented by design in this iteration

- No fine-tuning (intentionally excluded).
- No retrieval/OCR/KB (intentionally excluded).
- No Track A refactor (intentionally excluded).

### C) Optional future hardening (not required for first Track B)

- Add optional fallback model config (open-access) for immediate smoke runs.
- Add progress bar for long inference loops.
- Add optional truncation safeguards for very long generated text.
- Add explicit local model cache/offline option flags.

---

## 8) Exact runtime flow of Track B

1. CLI enters `src/train/eval_vlm_vqa.py`.
2. Loads `TrackBVLMEvalConfig` from JSON.
3. Applies optional subset override.
4. Resolves device and seeds.
5. Builds `VQAv2VLMEvalDataset` and DataLoader.
6. Constructs `PaliGemmaRunner`.
7. For each batch:
   - generate answers,
   - normalize predictions and gold,
   - compute per-sample exact and soft score,
   - collect rows.
8. Compute aggregate metrics:
   - top1-style normalized exact
   - VQA soft accuracy
   - answer-type breakdown
   - supplementary overlap metrics (optional)
9. Save JSONL + JSON artifacts.
10. Print summary.

---

## 9) Metric semantics in Track B (important)

- `vqa_soft_accuracy` remains **primary metric**.
- `top1_accuracy` in Track B is interpreted as:
  - normalized exact-match rate between generated answer and gold majority answer.

This is the closest equivalent to “top1-style exact” in generative evaluation.

- Answer-type breakdown computes exact and soft per type (`yes/no`, `number`, `other`) using existing utility logic.

---

## 10) Why this is the smallest clean implementation

This implementation intentionally avoids:
- touching classical model code,
- changing Track A datasets,
- inventing new metrics,
- introducing retrieval/OCR/KB complexity.

It adds only the minimum components required for Track B inference-first evaluation:

- one Track B dataset file,
- one VLM runner,
- one eval entrypoint,
- one Track B config,
- one dependency update.

This is surgical and aligned with project constraints.

---

## 11) Command reference

### Main Track B eval command

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.train.eval_vlm_vqa --config configs/track_b_paligemma_vqav2_eval.json --subset-val 64
```

### If gated-model access issue appears

```bash
huggingface-cli login
```

Then rerun the eval command.

---

## 12) Final status snapshot

- **Track B codepath:** implemented.
- **Track A isolation:** preserved.
- **Metrics/logging reuse:** done.
- **Artifact outputs:** implemented.
- **End-to-end execution:** blocked only by external HF gated model access.

This concludes the first clean Track B inference-only implementation pass.
