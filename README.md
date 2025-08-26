# Face Embedding Extraction & Clustering Pipeline

> **Composition:**

`01_extract.py` (embedding extraction) →&#x20;
`02_cluster.py` (FAISS‑based deduplication + clustering + label generation)

This repo takes a **list of image paths** as input, extracts face embeddings,&#x20;
builds a **FAISS** index to perform **duplicate removal** and **clustering**, and finally&#x20;
produces training files `classes.npy` / `data_list.npy`.

---

## 📁 Directory Layout (example)

```
project/
├─ 01_extract.py               
├─ 02_cluster.py
│  
├─ 01_input/
   └─ ai_model/
├─ 01_output/
   └─ current/
│  
├─ 02_input/
├─ 02_output/
   └─ current/
│  
└─ README.md
```

---

# 🧰 Requirements

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### GPU notes

> **Caution**
> `faiss-gpu` must match your CUDA/driver stack.

---

# 🚀 Quick Start (End‑to‑End)

1. **Extract embeddings**

```bash
python 01_extract.py \
  --input 01_input/5M_127M_CFSC_data_list.txt \
  --output 01_output/current \
  --weights /path/to/AVG_pruned_weight.pth \
  --batch-size 1024 \
  --num-workers 8 \
  --emb-dim 512 \
  --use-l2-norm
```

2. **Clustering + dedup + label generation**

```bash
python 02_cluster.py \
  --input 01_output/current \
  --output 02_output/current \
  --use_gpu \
  --remove_duplicates \
  --duplicate_thresh 0.90 \
  --cluster_thresh 0.70 \
  --topK 100 \
  --batch 2000000 \
  --save_groups 50
```

> `--batch` controls how many rows (samples) are processed per FAISS search chunk. Tune for your memory budget.

---

## 🧱 Stage 1: Embedding Extraction — `01_extract.py`

Generate embeddings from an image list using a pretrained model.&#x20;
Example list: `01_input/5M_127M_CFSC_data_list.txt`&#x20;
Preprocessing inside the script: `Resize(112×112) → ToTensor → Normalize`.

### Arguments

* `--input` *(str, default: `01_input/5M_127M_CFSC_data_list.txt`)* &#x20;
  Either a **folder** (recursively scans all `jpg/jpeg/png/bmp/webp`)&#x20;
  or a **txt file** containing one path per line.
* `--output` *(str, default: `./01_output/current`)* &#x20;
  Output root directory.
* `--weights` *(str)* &#x20;
  Path to `iresnet200_revised` weights (`.pth`).
* `--batch-size` *(int, default: `33768`)* &#x20;
  Batch size. **Use a realistic value for your GPU** (e.g., 512–4096).
* `--num-workers` *(int, default: `8`)* &#x20;
  DataLoader workers.
* `--emb-dim` *(int, `{512, 25600}`, default: `512`)*
* `--use-l2-norm` *(flag)* &#x20;
  Save L2‑normalized embeddings.
* `--no-merge-embs` *(flag)* &#x20;
  Save per‑batch temporary NPY files (debug/very large datasets).
* `--split-threshold` *(int, default: `1000000`)* &#x20;
  Split final output files when the number of samples exceeds this threshold.

### Output

* When merged (default):

  * `final_fpaths.npy` (list)
  * `final_indices.npy` (list; original input order indices)
  * `final_embs.npy` **or** `final_embs_norm.npy` (if `--use-l2-norm`)
* When not merged (`--no-merge-embs`):

  * `fpaths_0.npy`, `indices_0.npy`, `embs_0.npy` (or `embs_norm_0.npy`) …

> The model is wrapped with `torch.nn.DataParallel`, so **all available GPUs** will be used for inference.

---

## 🧭 Stage 2: Clustering — `02_cluster.py`

Using FAISS `IndexFlatIP` (equivalent to cosine similarity after L2‑norm), the pipeline performs:

* **STEP 1** Duplicate removal (connected components)
* **STEP 2** Same‑identity clustering (thresholded Union‑Find)
* **STEP 3** Group collection
  Then it generates training label files (`classes.npy` / `data_list.npy`).

### Required input folder structure

Results from `01_extract.py` must exist:
Either of:

* `final_fpaths.npy` + `final_embs.npy` (or `final_embs_norm.npy`)
* Sharded: `fpaths_*.npy` + `embs_*.npy` (or `embs_norm_*.npy`)

### Arguments

* `--input` *(str, default: `./02_input/extrac_full`)* &#x20;
  Folder containing the NPY files above.
* `--output` *(str, default: `./02_output/current`)* &#x20;
  Output root (labels / preview images / logs).
* `--use_gpu` *(flag, default: **True**)* &#x20;
  Attempt to build a sharded GPU index across **all GPUs** if available.
* `--remove_duplicates` *(flag, default: **True**)* &#x20;
  Enable duplicate removal.
* `--duplicate_thresh` *(float, default: `0.9`)* &#x20;
  Similarity threshold for duplicates.
* `--cluster_thresh` *(float, default: `0.7`)* &#x20;
  Similarity threshold for same‑identity clustering.
* `--topK` *(int, default: `100`)* &#x20;
  Number of neighbors per search.
* `--batch` *(int, default: `20000000`)* &#x20;
  Rows processed per search batch. Reduce for lower memory usage.
* `--save_groups` *(int, default: `50`)* &#x20;
  Save previews for the top‑N largest groups (max 10 images per group, resized to 112×112).

### Output

* **FAISS search & clustering logs**: console + file (`clustering_result.txt`)
* **Group previews**: `output/` → `group_0000/…` (when requested)
* **Training label files** *(auto‑generated)* under `output/`:

  * `fr_train_{X_class|XK_class}_{Y_data|YK_data}_classes.npy`
  * `fr_train_{...}_data_list.npy`
    e.g., `fr_train_12K_class_3K_data_classes.npy`

---

## 📊 Parameter Guide

* **Normalization & Metric**: In `02_cluster.py`, embeddings are L2‑normalized and searched with `IndexFlatIP` → effectively cosine similarity.
  (Even if you saved normalized embeddings in stage 1, stage 2 normalizes again.)
* **Thresholds**: Tune `duplicate_thresh` (\~0.9) and `cluster_thresh` (0.6–0.8) based on data quality/difficulty.
* **Confirm GPU index**: Look for the log line `Sharded index across N GPU(s)`.

---

## 🛠️ Troubleshooting

* **`CUBLAS_STATUS_EXECUTION_FAILED`**
  Likely CUDA/driver mismatch or OOM. Lower batch size, disable FP16, align versions.
* **`AxisError: axis 1 is out of bounds for array of dimension 1`**
  Input embeddings are not 2‑D. After loading, check `arr.shape`; enforce with `np.asarray(arr, dtype=np.float32)`.
* **GPU but no speed‑up**
  Ensure a **GPU index** is actually built; check `topK`/`nprobe` (if using IVF) and other bottlenecks.
* **Memory pressure**
  Reduce `--batch` (stage 2) and `--batch-size` (stage 1) for streaming.

---

## 🔁 Reproducibility & Logging Checklist

* [ ] Stage 1 outputs: `final_*.npy` or sharded `*_i.npy` pairs exist
* [ ] Stage 2 input folder contains both embedding & path files
* [ ] With `--remove_duplicates` ON, confirm removed count
* [ ] Verify pairing consistency between `classes.npy` and `data_list.npy`

---
### extract_example

### cluster_example
<img width="715" height="942" alt="cluster_example" src="https://github.com/user-attachments/assets/2a8bdc66-2dc7-4540-9957-ac43dbe0db29" />
