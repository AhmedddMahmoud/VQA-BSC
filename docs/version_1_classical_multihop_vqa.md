# Version 1 — Classical Multi-Hop VQA Baseline

## Purpose
Build a **classical, understandable VQA system** that answers questions from an image **without external knowledge**.

This version is not meant to be the final system. It is the control system that teaches you:
- how the VQA pipeline works end to end,
- how question encoding and visual attention interact,
- what the answer space looks like,
- what failure cases look like before adding knowledge or retrieval.

---

## Core Research Goal
**Can a classical multi-hop attention model solve standard VQA reasonably well, and how does it behave on a more reasoning-heavy benchmark?**

---

## Datasets

### Primary dataset: VQA v2
Use this as the main standard benchmark. The official VQA site says VQA v2 contains more than **250K images and 1.1M questions**, and was designed to reduce language bias compared with VQA v1. citeturn793722search15turn793722search11

- Official download: https://visualqa.org/download.html
- Main files you need:
  - questions
  - annotations
  - COCO train/val images

### Secondary dataset: GQA
Use this as the stress-test benchmark for stronger visual reasoning. The official GQA site states the dataset has **more than 110K images and 22M questions**, with scene graphs and question semantic programs. citeturn793722search12turn793722search5

- Official download: https://cs.stanford.edu/people/dorarad/gqa/download.html
- Start with:
  - questions
  - images
- Only download scene graphs if you need them later.

---

## Exact Model Choice

### Baseline you should implement
Use a **Stacked-Attention / Multi-Hop Attention style model**.

Recommended architecture:
1. **Image encoder**
   - Best practical option: pretrained **ResNet-50** feature map
   - Alternative: pretrained **ViT-B/16** if already comfortable with ViTs
2. **Question encoder**
   - **BiLSTM** or **GRU**
3. **Visual attention module**
   - 2 attention hops
   - question-conditioned spatial attention over image features
4. **Fusion**
   - combine attended image vector with question vector after each hop
5. **Answer head**
   - top-K answer classification

### Why this model
This is historically aligned with classical VQA and still light enough to implement and debug.

---

## Recommended Tensor-Level Design

### Image branch
- Input image resized to **224 × 224**
- Encoder: pretrained ResNet-50
- Use final convolutional map before global pooling
- Expected feature map shape:
  - `(B, 2048, 7, 7)`
- Flatten spatially to:
  - `(B, 49, 2048)`

### Question branch
- Tokenize question
- Build vocabulary from training split
- Encode with embedding layer + BiLSTM
- Use final hidden state or concatenated forward/backward hidden states
- Suggested question embedding size:
  - `512`

### Hop 1 attention
For each region feature `v_i` and question vector `q`:
- project both into a joint space
- score each spatial location
- apply softmax over 49 locations
- compute attended vector `v^(1)`

Suggested equations:
- `h_i = tanh(W_v v_i + W_q q + b)`
- `a_i = w^T h_i`
- `alpha = softmax(a)`
- `v_att = sum(alpha_i * v_i)`

### Hop update
- `u^(1) = q + W_1 v_att`

### Hop 2 attention
Repeat attention using `u^(1)` instead of `q`.

### Final classifier
- `z = concat(u^(1), v_att^(2))` or `z = u^(2)` depending on your implementation
- MLP → logits over answer vocabulary

---

## Answer Space Rules
Use **top-K most frequent answers** from the training set.

Recommended:
- start with **top 1000** answers,
- if stable, move to **top 3000**.

Why:
- easier training,
- smaller classifier,
- manageable laptop memory.

Do not try full open-vocabulary generation in Version 1.

---

## Training Objective
Use **multi-class classification** over the answer vocabulary.

Recommended loss:
- Cross-entropy for the first working baseline.

Better but optional:
- soft target scoring based on VQA answer frequency.

---

## Evaluation

### On VQA v2
Use standard VQA accuracy evaluation later if possible, but for your first training loop you can monitor:
- top-1 classifier accuracy on val subset,
- answer-type breakdown if available.

### On GQA
Track:
- validation accuracy,
- compare to VQA v2 behavior,
- note failures on compositional/reasoning-heavy questions.

---

## What to Do on a Laptop

### If your laptop is limited
Do **not** train on the full dataset first.

Use this fallback plan:
1. Train on a **small subset** first:
   - 10k–30k VQA v2 train samples
   - 5k validation samples
2. Use **pre-extracted image features** if needed
3. Freeze the image encoder initially
4. Train only:
   - question encoder
   - attention layers
   - classifier head

### Best laptop-safe configuration
- ResNet-50 frozen
- 224 × 224 images
- batch size 16 or 32
- mixed precision if available
- top-1000 answers
- 2 attention hops

### If even that is too heavy
Use one of these alternatives:

#### Alternative A — Precompute features once
Extract and save `(49, 2048)` features per image, then train only the VQA head.

#### Alternative B — Use BLIP VQA as a comparison baseline instead of training your own heavy image branch
Hugging Face provides `Salesforce/blip-vqa-base`, and BLIP is designed for visual-language understanding and generation. citeturn686075search1turn686075search5

Use your custom multi-hop model as the **educational baseline**, and BLIP as the **strong practical reference baseline**.

---

## Models to Use

### Must-implement model
- Custom multi-hop attention VQA model

### Strong reference model
- `Salesforce/blip-vqa-base` via Hugging Face or LAVIS. BLIP is documented by Hugging Face, and LAVIS provides an official language-vision research library from Salesforce. citeturn686075search1turn686075search0

### Why keep BLIP around
Because if your custom baseline underperforms badly, you still have a strong comparison point.

---

## Libraries
Use:
- `PyTorch`
- `torchvision`
- `transformers`
- `tqdm`
- `numpy`
- `pandas`
- optionally `LAVIS`

---

## Repo Structure

```text
project/
  data/
    vqa_v2/
    gqa/
  notebooks/
  src/
    datasets/
      vqa_dataset.py
      gqa_dataset.py
    models/
      image_encoder.py
      question_encoder.py
      attention.py
      multihop_vqa.py
    train/
      train_vqa.py
      eval_vqa.py
    utils/
      vocab.py
      metrics.py
      config.py
  outputs/
    checkpoints/
    logs/
    predictions/
```

---

## Rules You Must Follow
1. **Do not overcomplicate Version 1.** No retrieval, no OCR, no external knowledge.
2. **Get one clean end-to-end run first.**
3. **Use a small subset before full training.**
4. **Keep the image encoder frozen at first.**
5. **Document every decision**:
   - answer vocabulary size,
   - subset size,
   - feature shape,
   - number of hops,
   - hidden dimensions.
6. **Save qualitative failures** early.
7. **Do not chase SOTA.** The goal is a clean baseline.

---

## Success Criteria
Version 1 is successful if:
- it trains end to end,
- it answers standard VQA questions non-trivially,
- you can compare behavior on VQA v2 vs GQA,
- you understand exactly where it fails.

---

## Output of Version 1
By the end of this stage you should have:
- a working classical multi-hop VQA model,
- one strong reference baseline (BLIP),
- train/val evaluation scripts,
- qualitative error examples,
- a clean foundation for adding knowledge later.

---

## Current Status (March 2026)

### Completed
- [x] Classical multi-hop VQA baseline implemented (frozen `ResNet-50` + `GRU/BiLSTM` + 2-hop attention + top-K classifier).
- [x] VQA v2 data pipeline implemented with answer filtering and subset controls.
- [x] Train/eval scripts implemented.
- [x] Debug utilities implemented:
  - `inspect_vqa_samples.py`
  - `debug_forward_pass.py`
  - `overfit_small_batch.py`
- [x] Attention weights are returned for debugging.
- [x] Qualitative failure logging is implemented.
- [x] Tiny-subset overfit behavior has been demonstrated with saved predictions/checkpoint artifacts.

### Still Missing / Not Finished Yet
- [ ] One logged, clean **small real train+val run** completion report (multi-batch train + val + checkpoint + qualitative failures from main training script).
- [ ] GQA stress-test comparison stage.
- [ ] Strong reference baseline execution/comparison (BLIP) for side-by-side analysis.
- [ ] Final consolidated analysis report comparing:
  - VQA v2 behavior,
  - GQA behavior,
  - failure modes and limitations.

### Immediate Next Step
- Run a small real-data training proof run (for example, train subset 256 and val subset 64), verify:
  - end-to-end training loop,
  - validation pass,
  - checkpoint save,
  - qualitative failures save,
  - no path/shape/runtime errors.
