🐯 Tiger Biometric Re-ID — Pure ML Algorithmic Pipeline Audit

> **Audit Date:** October 2026  
> **Project Root:** `d:\test-ONNX-model`  
> **Dataset Root:** `d:\test-ONNX-model\Tadoba Dataset\`

---

## 📋 Table of Contents

1. [Overall Status Summary](#1-overall-status-summary)
2. [File Inventory & Purpose](#2-file-inventory--purpose)
3. [Stage-by-Stage Pipeline Audit](#3-stage-by-stage-pipeline-audit)
4. [ML Models: Train Now vs. Train Later](#4-ml-models-train-now-vs-train-later)
5. [Critical Gaps & Bugs Found](#5-critical-gaps--bugs-found)
6. [Model Artifacts Status](#6-model-artifacts-status)
7. [Path Issues That Break Portability](#7-path-issues-that-break-portability)
8. [What Must Be Built Next](#8-what-must-be-built-next)

---

## 1. Overall Status Summary

| Pipeline Layer | Architecture Requirement | Implementation Status |
|---|---|---|
| **Stage 1:** YOLO Localization | YOLOv8 → BBox `[x1,y1,x2,y2]` | ✅ Implemented (`quality_gate_pipeline.py`) |
| **Stage 2:** IQA Blur Filter | Laplacian Variance `σ² < 100` → Reject | ✅ Implemented (`quality_gate_pipeline.py`) |
| **Stage 3:** Tensor Normalization | Resize `128×256`, ImageNet norm, RandomGrayscale | ✅ Implemented (training + inference) |
| **Stage 4:** ResNet-50 Backbone | Layers 1–4 feature extraction | ✅ Implemented (`TigerPPbM_ResNet50`) |
| **Stage 5:** PPbM-b Attention (Core) | Part pooling `(1,4)` + SE Channel Attention | ✅ Implemented (both train files) |
| **Stage 6:** L2 Normalization → 512-D | Bottleneck → `F.normalize(p=2)` | ✅ Implemented |
| **Stage 7:** Cosine Similarity Search | Gallery DB query, threshold ≥ 0.75 | ❌ NOT IMPLEMENTED |
| **Flank Classifier** | Left (0) / Right (1) orientation routing | ❌ NOT IMPLEMENTED |
| **ONNX Export (Re-ID)** | `tiger_reid.onnx`, opset 18 | ✅ Script exists, ⚠️ wrong path |
| **ONNX Export (YOLOv8)** | `yolov8n.onnx` for offline deploy | ❌ NOT IMPLEMENTED |
| **TFLite Mobile Targets** | `yolov8n.tflite`, `tiger_reid_fp16.tflite` | ❌ NOT IMPLEMENTED |

---

## 2. File Inventory & Purpose

### Root: `d:\test-ONNX-model\`

| File | Role | Status |
|---|---|---|
| `train_local.py` | Quick 5-epoch verification training | ✅ Functional |
| `train_full.py` | Production 60-epoch training | ✅ Functional, already ran |
| `export_onnx.py` | PyTorch → ONNX export (opset 18) | ⚠️ Wrong hardcoded path |
| `quality_gate_pipeline.py` | YOLO + Blur + Re-ID inference pipeline | ⚠️ Wrong hardcoded path, no gallery DB |
| `yolov8n.pt` | Stock COCO YOLOv8n checkpoint | ✅ Present (6.54 MB) |
| `requirements.txt` | Python dependencies | ✅ Present |
| `debug_1_tight_crop.jpg` | Crop visualization output | ✅ Generated |
| `debug_2_yolo_detection.jpg` | BBox overlay visualization | ✅ Generated |

### Weights: `Tadoba Dataset\weights\`

| File | Size | Status |
|---|---|---|
| `tiger_reid_epoch_10.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_epoch_20.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_epoch_30.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_epoch_40.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_epoch_50.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_epoch_60.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid_final.pth` | 111.9 MB | ✅ Exists |
| `tiger_reid.onnx` | Expected in weights/ | ❌ MISSING — path mismatch |

---

## 3. Stage-by-Stage Pipeline Audit

---

### Stage 1 — Raw Tensor Ingestion & Localization (YOLOv8) ✅

**Required:** YOLOv8 CNN processes `1920×1080` raw frame → BBox `[x1, y1, x2, y2]`

**Implementation in `quality_gate_pipeline.py` (L48–69):**

```python
results = self.detector(img_bgr, verbose=False)[0]          # YOLOv8 forward pass
best_box = max(boxes, key=lambda b: float(b.conf[0]))       # Highest confidence
x1, y1, x2, y2 = [clamped coordinates]                     # BBox extraction
crop = img_bgr[y1:y2, x1:x2]                               # Crop the animal
```

**Verdict:** ✅ Correctly implemented.
**Note:** Uses `yolov8n.pt` (stock COCO model). Not fine-tuned for tigers/camera traps. Human rejection via `cls_id == 0` is correct for COCO.

---

### Stage 2 — Deterministic Image Quality Assessment (IQA) ✅

**Required:** Laplacian filter → `σ²_Lap = Var(∇²I)`. Reject if `σ² < 100`

**Implementation in `quality_gate_pipeline.py` (L21–25):**

```python
def check_blur(self, crop_bgr, threshold=100.0):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()    # σ²_Lap
    return variance >= threshold, variance               # Gate: reject if < 100
```

**Verdict:** ✅ Formula matches specification exactly. Threshold = 100.

---

### Stage 3 — Biological Tensor Normalization ✅

**Required:** Resize to `128×256` (2:1 quadruped ratio), ImageNet normalization, `RandomGrayscale(p=0.3)` for IR/day-camera invariance

**Training — `train_full.py` (L41–47):**
```python
T.Resize((128, 256)),
T.RandomGrayscale(p=0.3),       # ✅ IR/Day invariance
T.ColorJitter(brightness=0.2, contrast=0.2),
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Inference — `quality_gate_pipeline.py` (L27–38):**
```python
resized = cv2.resize(crop_rgb, (256, 128))   # ✅ width=256, height=128 → [128×256] tensor
tensor = (tensor - mean) / std               # ✅ ImageNet normalization
tensor = np.transpose(tensor, (2, 0, 1))    # ✅ HWC → CHW
np.expand_dims(tensor, axis=0)              # ✅ → [1, 3, 128, 256]
```

**Verdict:** ✅ Fully aligned. `RandomGrayscale` forces stripe-edge learning, not color.

---

### Stage 4 — Hierarchical Feature Extraction (ResNet-50 Backbone) ✅

**Required:** Normalized tensor `[1, 3, 128, 256]` through ResNet-50 Layers 1→4

**Implementation in `train_full.py` (L107–118):**

```python
x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # Stem
x = self.layer2(self.layer1(x))                        # Layers 1 & 2
feat_l3 = self.layer3(x)                               # Layer 3 → 1024-ch feature maps
feat_l4 = self.layer4(feat_l3)                         # Layer 4 → 2048-ch feature maps
```

**Verdict:** ✅ Standard ResNet-50 backbone, all 4 layers connected correctly.

---

### Stage 5 — Pose Part-Based Attention (PPbM-b) ✅ — CORE INNOVATION

**Required:**
- `AdaptiveAvgPool2d((1, 4))` at `res3d` → 4 horizontal stripe bands (Forequarters / Anterior Flank / Posterior Flank / Rump)
- SE block: `α_i` via `amax + mean` pooling → suppress occluded body parts

**Part Pooling — `train_full.py` (L97–113):**
```python
self.part_pool = nn.AdaptiveAvgPool2d((1, 4))           # ✅ 4 horizontal stripe bands
part_emb = F.relu(self.part_proj(
    self.part_pool(self.ca_layer3(feat_l3) * feat_l3)  # ✅ Attention-weighted part features
    .flatten(1)))
```

**Channel Attention (SE Block) — `train_full.py` (L79–88):**
```python
class ChannelAttention(nn.Module):
    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(x.mean(dim=[-2, -1], keepdim=True))))  # ✅ Mean pool
        max_out = self.fc2(F.relu(self.fc1(x.amax(dim=[-2, -1], keepdim=True)))) # ✅ amax pool
        return torch.sigmoid(avg_out + max_out)                                    # ✅ α_i weights
```

> **Minor discrepancy:** `train_local.py` uses `F.adaptive_avg_pool2d` / `F.adaptive_max_pool2d` for SE pooling, while `train_full.py` (production) uses the correct `x.mean()` / `x.amax()` approach matching the spec. The **production script is correct**.

**Verdict:** ✅ Fully implemented. The `(1,4)` pooling creates the 4 stripe columns as specified.

---

### Stage 6 — Dimensionality Reduction & L2 Normalization ✅

**Required:** `[global_emb (512) + part_emb (512)]` → Bottleneck `Linear(1024→512)` → `F.normalize(p=2)`

**Implementation — `train_full.py` (L104–117):**
```python
self.bottleneck = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512))

# Forward:
norm_emb = F.normalize(
    self.bottleneck(torch.cat([global_emb, part_emb], dim=1)),  # 512+512=1024 → 512
    p=2, dim=1                                                   # ✅ L2 onto unit hypersphere
)
```

**Verdict:** ✅ Exactly matches architecture. Output is a 512-D L2-normalized unit vector.

---

### Stage 7 — Open-Set Metric Search (Cosine Similarity) ❌ NOT IMPLEMENTED

**Required:** For query `q` and gallery `{g_k}`:
- `S(q, g_k) = Σ q_i · g_{k,i}` (dot product = cosine sim since both L2-normalized)
- **Match:** `S ≥ 0.75` → Known tiger ID
- **Uncertain:** `0.65 ≤ S < 0.75` → Flag for manual review
- **Unknown:** `S < 0.65` → New tiger, create new identity

**Current state of `quality_gate_pipeline.py` (L86–97):**
```python
# Pipeline STOPS HERE — returns raw embedding, no DB query
outputs = self.reid_session.run(None, {self.input_name: reid_input})
embedding_512 = outputs[0][0]
return {"status": "ACCEPTED", "embedding": embedding_512}  # ← No gallery search!
```

**Verdict:** ❌ This is the single biggest missing piece. The 512-D vector is extracted but never compared against any gallery. There is no SQLite or FAISS database, no cosine similarity search, and no identity decision logic.

---

### Flank Orientation Classifier ❌ NOT IMPLEMENTED

**Required:** Because tiger flank stripes are asymmetrical, a MobileNetV3 binary classifier must sort each crop into:
- `Class 0` → Left flank → Query left-flank gallery index
- `Class 1` → Right flank → Query right-flank gallery index

**Current state:** No script, no model, no training data split exists for this classifier.

**Verdict:** ❌ Completely missing. Without this, left and right flank embeddings are mixed in the same gallery, artificially reducing matching accuracy.

---

## 4. ML Models: Train Now vs. Train Later

### 🔴 TRAIN NOW — Cannot Be Deferred

| Model | Why Train Now | What You Need |
|---|---|---|
| **Flank Orientation Classifier (MobileNetV3-Small)** | Every image processed by the pipeline needs its embedding routed to the correct left/right gallery. Without this, Stage 7 cosine search is unreliable. Training is fast (2–4 hours on GTX 1650). | Labeled left/right flank images from ATRW reid_train set. Train using `train_flank_classifier.py` (to be created). Output: `flank_classifier.onnx` |
| **ONNX Export Fix (Re-ID model)** | `export_onnx.py` has wrong hardcoded path (`D:\Tadoba_Dataset\weights\`) but weights are at `D:\test-ONNX-model\Tadoba Dataset\weights\`. The ONNX file does not exist at the path `quality_gate_pipeline.py` reads from. | Fix path, re-run `export_onnx.py`. Takes < 2 minutes. |

### 🟡 TRAIN LATER — Can Be Deferred to Next Phase

| Model | Why It Can Wait | Notes |
|---|---|---|
| **Fine-tuned YOLOv8 for Tigers** | Stock COCO model detects 4-legged animals well enough for prototype. | Requires tiger-specific bbox annotation dataset. Not in ATRW reid split. |
| **YOLOv8 ONNX Export** | `.pt` format runs fine on workstation. ONNX only needed for cross-platform offline deploy. | `yolo export model=yolov8n.pt format=onnx imgsz=640` — 1 command, 5 minutes. |
| **TFLite Mobile Models** | Android Ranger companion app is a future phase. Not needed for core workstation demo. | `tiger_reid_fp16.tflite`, `yolov8n.tflite` — quantized exports. |

---

## 5. Critical Gaps & Bugs Found

### Bug 1 — Hardcoded Path Mismatch (Breaks Portability) ⚠️

| File | Hardcoded Path (Wrong) | Actual Path |
|---|---|---|
| `export_onnx.py` L11 | `D:\Tadoba_Dataset\weights\` | `D:\test-ONNX-model\Tadoba Dataset\weights\` |
| `quality_gate_pipeline.py` L8 | `D:\Tadoba_Dataset\` | `D:\test-ONNX-model\Tadoba Dataset\` |
| `quality_gate_pipeline.py` L10 | `train\train\002732.jpg` | Check actual ATRW folder structure |

**Fix:** Replace all hardcoded absolute paths with relative paths using `Path(__file__).parent`.

---

### Bug 2 — ONNX File Does Not Exist at the Expected Path ❌

`quality_gate_pipeline.py` tries to load:
```
D:\Tadoba_Dataset\weights\tiger_reid.onnx
```
This path does not exist. The weights folder at `D:\test-ONNX-model\Tadoba Dataset\weights\` contains only `.pth` checkpoints — **no `.onnx` file**.

`export_onnx.py` must be run with the corrected path to generate it.

---

### Bug 3 — Minor SE Block Inconsistency Between Scripts

`train_local.py` uses:
```python
avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
```

`train_full.py` (production) uses:
```python
avg_out = self.fc2(F.relu(self.fc1(x.mean(dim=[-2, -1], keepdim=True))))
max_out = self.fc2(F.relu(self.fc1(x.amax(dim=[-2, -1], keepdim=True))))
```

Both are mathematically equivalent, but `train_full.py` matches the architecture spec more precisely (uses `amax` as specified). **The production `train_full.py` is the canonical version.**

---

### Gap 4 — No Gallery Database Exists ❌

There is no:
- SQLite file for tiger identity records
- FAISS index file for approximate nearest-neighbor search
- Gallery embedding file (`.npy` matrix of known tiger vectors)
- Any code to build, populate, or query a gallery

**This means even if `quality_gate_pipeline.py` runs successfully, the system cannot identify any tiger — it can only extract the 512-D vector and return it.**

---

### Gap 5 — No Flank Classifier Script or Model ❌

The architecture requires routing embeddings to separate left/right flank indexes. No script exists to:
1. Split the ATRW reid training images by flank orientation
2. Train a `MobileNetV3-Small` binary classifier
3. Export it to ONNX

---

## 6. Model Artifacts Status

| Artifact | Expected Location | Actual Status |
|---|---|---|
| `tiger_reid_final.pth` | `Tadoba Dataset\weights\` | ✅ EXISTS (111.9 MB) |
| `tiger_reid_epoch_*.pth` | `Tadoba Dataset\weights\` | ✅ EXISTS (×6, epochs 10–60) |
| `tiger_reid.onnx` | `Tadoba Dataset\weights\` | ❌ MISSING — `export_onnx.py` not yet run with correct path |
| `yolov8n.pt` | Project root | ✅ EXISTS (6.54 MB, COCO pretrained) |
| `yolov8n.onnx` | — | ❌ MISSING |
| `flank_classifier.pth` / `.onnx` | — | ❌ MISSING (not trained) |
| `yolov8n.tflite` | — | ❌ MISSING |
| `tiger_reid_fp16.tflite` | — | ❌ MISSING |
| `tiger_gallery.db` (SQLite) | — | ❌ MISSING (not built) |
| `tiger_gallery.faiss` | — | ❌ MISSING (not built) |

---

## 7. Path Issues That Break Portability

All four Python files use absolute paths that only work on this specific machine's `D:` drive layout.

### Recommended Fix Pattern

Replace in all scripts:
```python
# CURRENT (broken on any other machine)
DATA_ROOT = Path(r"D:\Tadoba_Dataset")

# FIXED (relative to this script's location)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Tadoba Dataset"
WEIGHTS_DIR = DATA_ROOT / "weights"
ONNX_PATH = WEIGHTS_DIR / "tiger_reid.onnx"
```

### Files Needing Path Fixes

| File | Lines to Fix |
|---|---|
| `export_onnx.py` | L11–13: `DATA_ROOT`, `weights_path`, `onnx_output` |
| `quality_gate_pipeline.py` | L8–10: `DATA_ROOT`, `ONNX_PATH`, `SAMPLE_IMG` |
| `train_local.py` | L16–18: `DATA_ROOT`, `IMG_DIR`, `CSV_PATH` |
| `train_full.py` | L19–25: `DATA_ROOT`, `IMG_DIR`, `CSV_PATH`, `WEIGHTS_DIR` |

---

## 8. What Must Be Built Next

### Priority 1 — Fix & Complete (Immediate)

```
[ ] Fix hardcoded paths in all 4 scripts → use Path(__file__).parent
[ ] Re-run export_onnx.py with corrected path → generates tiger_reid.onnx
[ ] Verify quality_gate_pipeline.py loads the ONNX correctly end-to-end
```

### Priority 2 — Stage 7: Gallery Database & Cosine Search

Create `build_gallery.py`:
```python
# Build tiger gallery embeddings from ATRW reid train set
# Run quality_gate_pipeline for each known tiger image
# Store: { tiger_id: embedding_512 } → tiger_gallery.npy + tiger_ids.json
```

Create `identify_tiger.py` (cosine similarity matcher):
```python
def identify_tiger(query_embedding, gallery_embeddings, gallery_ids, threshold=0.75):
    scores = np.dot(gallery_embeddings, query_embedding)  # Cosine sim (L2-normalized)
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score >= 0.75:
        return "MATCH", gallery_ids[best_idx], best_score
    elif best_score >= 0.65:
        return "UNCERTAIN", gallery_ids[best_idx], best_score
    else:
        return "UNKNOWN_NEW_TIGER", None, best_score
```

### Priority 3 — Flank Orientation Classifier

Create `train_flank_classifier.py`:
```python
# MobileNetV3-Small, binary output (left=0, right=1)
# Input: 224x224 RGB crop
# Training data: manually labeled subset of ATRW reid_train
# Export: flank_classifier.onnx
```

### Priority 4 — Export YOLOv8 to ONNX (1 Command, 5 Minutes)

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640
```

---

## Architecture Compliance Score

| Component | Required | Implemented | Score |
|---|---|---|---|
| YOLOv8 Detection | ✅ | ✅ | 100% |
| Laplacian IQA Filter | ✅ | ✅ | 100% |
| 128×256 Normalization + RandomGrayscale | ✅ | ✅ | 100% |
| ResNet-50 Backbone (Layers 1–4) | ✅ | ✅ | 100% |
| Part Pooling `AdaptiveAvgPool2d((1,4))` | ✅ | ✅ | 100% |
| SE Channel Attention (amax + mean) | ✅ | ✅ | 100% |
| 512-D Bottleneck + L2 Normalize | ✅ | ✅ | 100% |
| TriHard Loss (margin=0.3) | ✅ | ✅ | 100% |
| Cosine Similarity Gallery Search | ✅ | ❌ | 0% |
| Threshold Decision (0.65/0.75) | ✅ | ❌ | 0% |
| Flank Orientation Classifier | ✅ | ❌ | 0% |
| ONNX Re-ID Export | ✅ | ⚠️ Path bug | 50% |
| YOLOv8 ONNX Export | ✅ | ❌ | 0% |

**Overall Pipeline Completeness: 7 / 13 stages fully implemented (~54%)**

> **IMPORTANT:** The Biometric Core (Stages 1–6) is 100% correctly implemented and the model weights are fully trained (60 epochs, `tiger_reid_final.pth` = 111.9 MB). The project is blocked only on **Stage 7** (gallery search) and the **path bug** that prevents the ONNX file from loading.

> **WARNING — DO NOT RETRAIN:** `tiger_reid_final.pth` already exists. Re-training would waste hours of GPU time for no gain. The **only training task remaining** is the **Flank Classifier** (MobileNetV3-Small, fast < 4 hrs).

---

*This audit cross-references the stated architecture specification against actual source code in `train_local.py`, `train_full.py`, `export_onnx.py`, and `quality_gate_pipeline.py`, plus the file system contents of `Tadoba Dataset\weights\` and `Tadoba Dataset\Annotations\`.*
