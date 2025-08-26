import argparse
import os
import shutil
import numpy as np
import faiss
import cv2
from tqdm import tqdm
from math import ceil
import glob
import time
import re


def natural_sort_key(filename):
    numbers = re.findall(r'\d+', os.path.basename(filename))
    return [int(num) for num in numbers] if numbers else [0]


def load_labels_from_folder(folder_path):
    """
    Load and concatenate all classes.npy and data_list.npy files from specified folder.
    """
    print(f"Loading files from folder: {folder_path}")
    start_time = time.time()

    # Find all npy files in the folder
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    print(f"Found {len(npy_files)} .npy files")

    # Separate classes and data_list files
    embs_files = []
    fpaths_files = []

    for file in npy_files:
        filename = os.path.basename(file)
        if 'embs' in filename:
            embs_files.append(file)
        elif 'fpaths' in filename or 'list' in filename:
            fpaths_files.append(file)

    # Sort to ensure consistent order
    embs_files.sort(key=natural_sort_key)
    fpaths_files.sort(key=natural_sort_key)

    print(f"Embedding files found: {len(embs_files)}")
    for i, f in enumerate(embs_files):
        print(f"  {i+1}. {os.path.basename(f)}")

    print(f"Filepath files found: {len(fpaths_files)}")
    for i, f in enumerate(fpaths_files):
        print(f"  {i+1}. {os.path.basename(f)}")

    # Load embedding files
    print("\nLoading embedding files...")
    all_classes = []
    for i, classes_file in enumerate(embs_files):
        print(f"Loading embedding file {i+1}/{len(embs_files)}: {os.path.basename(classes_file)}")
        file_start = time.time()
        classes = np.load(classes_file)
        load_time = time.time() - file_start
        print(f"  Shape: {classes.shape}, Size: {classes.nbytes / 1024**2:.1f} MB, Time: {load_time:.1f}s")
        all_classes.append(classes)

    # Load filepath files
    print("\nLoading filepath files...")
    all_data_lists = []
    for i, data_file in enumerate(fpaths_files):
        print(f"Loading filepath file {i+1}/{len(fpaths_files)}: {os.path.basename(data_file)}")
        file_start = time.time()
        data_list = np.load(data_file, allow_pickle=True)
        load_time = time.time() - file_start
        print(f"  Length: {len(data_list)}, Time: {load_time:.1f}s")
        all_data_lists.append(data_list)

    # Concatenate all
    print("\nConcatenating arrays...")
    concat_start = time.time()

    print("  Concatenating embeddings...")
    concat_emb_start = time.time()

    if all_classes:
        total_rows = sum(arr.shape[0] for arr in all_classes)
        emb_dim = all_classes[0].shape[1]
        print(f"    Pre-allocating array for {total_rows:,} × {emb_dim} embeddings...")

        combined_embs = np.empty((total_rows, emb_dim), dtype=all_classes[0].dtype)

        current_idx = 0
        for i, arr in enumerate(all_classes):
            end_idx = current_idx + arr.shape[0]
            print(f"    Copying embedding array {i+1}/{len(all_classes)}: {arr.shape[0]:,} rows")
            combined_embs[current_idx:end_idx] = arr
            current_idx = end_idx

        emb_concat_time = time.time() - concat_emb_start
        print(f"    Embeddings concatenated: {combined_embs.shape}, Time: {emb_concat_time:.1f}s")
    else:
        combined_embs = np.array([])
        emb_concat_time = 0.0

    print("  Concatenating filepaths...")
    fpath_start = time.time()

    if all_data_lists:
        total_items = sum(len(arr) for arr in all_data_lists)
        print(f"    Pre-allocating array for {total_items:,} filepaths...")

        combined_fpaths_list = []
        for i, arr in enumerate(all_data_lists):
            print(f"    Copying filepath array {i+1}/{len(all_data_lists)}: {len(arr):,} items")
            combined_fpaths_list.extend(arr)

        combined_fpaths = np.array(combined_fpaths_list)
        fpath_concat_time = time.time() - fpath_start
        print(f"    Filepaths concatenated: {len(combined_fpaths)}, Time: {fpath_concat_time:.1f}s")
    else:
        combined_fpaths = np.array([])
        fpath_concat_time = 0.0

    total_time = time.time() - start_time
    print(f"\nFile loading completed in {total_time:.1f}s")
    print(f"Final embeddings shape: {combined_embs.shape}")
    print(f"Final filepath count: {len(combined_fpaths)}")

    return combined_embs, combined_fpaths


def load_embeddings(emb_paths):
    print(f"Loading {len(emb_paths)} embedding file(s)...")
    arrays = []
    for i, path in enumerate(emb_paths):
        print(f"  Loading embedding file {i+1}/{len(emb_paths)}: {os.path.basename(path)}")
        start_time = time.time()
        arr = np.load(path)
        load_time = time.time() - start_time
        print(f"    Shape: {arr.shape}, Size: {arr.nbytes / 1024**2:.1f} MB, Time: {load_time:.1f}s")
        arrays.append(arr)

    print("Concatenating embedding arrays...")
    start_time = time.time()
    result = np.concatenate(arrays, axis=0)
    concat_time = time.time() - start_time
    print(f"Final embedding shape: {result.shape}, Time: {concat_time:.1f}s")
    return result


def load_filepaths(fpath_paths):
    print(f"Loading {len(fpath_paths)} filepath file(s)...")
    arrays = []
    for i, path in enumerate(fpath_paths):
        print(f"  Loading filepath file {i+1}/{len(fpath_paths)}: {os.path.basename(path)}")
        start_time = time.time()
        arr = np.load(path, allow_pickle=True)
        load_time = time.time() - start_time
        print(f"    Length: {len(arr)}, Time: {load_time:.1f}s")
        arrays.append(arr)

    print("Concatenating filepath arrays...")
    start_time = time.time()
    result = np.concatenate(arrays, axis=0)
    concat_time = time.time() - start_time
    print(f"Final filepath count: {len(result)}, Time: {concat_time:.1f}s")
    return result


def build_faiss_index(embs, use_gpu=True, use_float16=False):
    print(f"Starting FAISS index building...")
    start_time = time.time()

    # 1) Force a true float32, C-contiguous ndarray
    print("  Step 1/4: Converting to float32...")
    embs = np.asarray(embs, dtype=np.float32, order='C')

    # 2) Normalize
    print("  Step 2/4: Normalizing embeddings...")
    norm_start = time.time()
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_norm = embs / norms
    embs_norm = np.asarray(embs_norm, dtype=np.float32, order='C')
    norm_time = time.time() - norm_start
    print(f"    Normalization completed in {norm_time:.1f}s")

    n, dim = embs_norm.shape
    print(f"  Building FAISS index on {n:,} vectors (dim={dim})...")

    # 3) Build the CPU index
    print("  Step 3/4: Building CPU index...")
    cpu_start = time.time()
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embs_norm)
    cpu_time = time.time() - cpu_start
    print(f"    CPU index built in {cpu_time:.1f}s")

    # 4) Try GPU only if FAISS has the GPU classes
    print("  Step 4/4: Setting up GPU index...")
    index = cpu_index
    if use_gpu and hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_all_gpus'):
        try:
            gpu_count = faiss.get_num_gpus()
        except Exception:
            gpu_count = 0
            print("GPU not working")
        if gpu_count > 0:
            try:
                gpu_start = time.time()
                res = faiss.StandardGpuResources()
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = use_float16
                index = faiss.index_cpu_to_all_gpus(cpu_index, co)
                gpu_time = time.time() - gpu_start
                print(f"    Sharded index across {gpu_count} GPU(s) in {gpu_time:.1f}s")
            except Exception as e:
                print(f"    GPU indexing failed ({e}); continuing with CPU index.")
        else:
            print("    No GPUs detected; using CPU index.")
    elif use_gpu:
        print("    FAISS not compiled with GPU support; using CPU index.")
    else:
        print("    GPU disabled; using CPU index.")

    total_time = time.time() - start_time
    print(f"FAISS index building completed in {total_time:.1f}s")
    return index, embs_norm


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


def find_duplicate_images(sims, idxs, duplicate_thresh, n, *, prefer_gpu=True, chunk_rows=200_000, keep_policy="min"):
    """
    기존: 파이썬 이중 for로 whiteset/blackset.
    변경: (배치) 벡터화 → (i,j) 에지 생성 → Union-Find 연결요소에서 하나만 keep.
    - 입력/출력 시그니처 동일 (추가 인자는 키워드 전용)
    - 메모리 큰 sims/idxs를 '그대로' 받는 상황을 가정하여, GPU로도 chunk 처리.

    keep_policy: 중복 컴포넌트에서 남길 대표 인덱스 선택 기준 ("min"|"max"|"first")
    """
    from math import ceil
    import numpy as np

    topK = idxs.shape[1]
    if sims.dtype != np.float32:
        sims = sims.astype(np.float32, copy=False)
    if idxs.dtype != np.int64:
        idxs = idxs.astype(np.int64, copy=False)

    # 내부 유니온-파인드 (기존 클래스 사용)
    uf = UnionFind(n)

    def _keep_blackset_from_components(uf_obj, total_n):
        comps = {}
        for t in range(total_n):
            r = uf_obj.find(t)
            comps.setdefault(r, []).append(t)
        black = set()
        for nodes in comps.values():
            if len(nodes) <= 1:
                continue
            if keep_policy == "max":
                keep = max(nodes)
            elif keep_policy == "first":
                keep = nodes[0]
            else:  # "min"
                keep = min(nodes)
            for v in nodes:
                if v != keep:
                    black.add(v)
        return black

    use_gpu = bool(_TORCH_OK and prefer_gpu and torch.cuda.is_available())
    batches = ceil(n / chunk_rows)
    print(f"[dup] n={n:,}, topK={topK}, chunk_rows={chunk_rows:,}, batches={batches}, gpu={use_gpu}")

    for bi, s in enumerate(range(0, n, chunk_rows), start=1):
        e = min(s + chunk_rows, n)

        if use_gpu:
            device = "cuda"
            Dt = torch.from_numpy(sims[s:e]).to(device=device, dtype=torch.float32, non_blocking=True)
            It = torch.from_numpy(idxs[s:e]).to(device=device, dtype=torch.int64,   non_blocking=True)
            row = torch.arange(s, e, device=device, dtype=torch.int64).unsqueeze(1).expand(e - s, topK)
            mask = (Dt >= duplicate_thresh) & (It != row)
            if mask.any():
                src = row[mask]
                dst = It[mask]
                a = torch.minimum(src, dst)
                b = torch.maximum(src, dst)
                edges = torch.stack((a, b), dim=1)
                edges = torch.unique(edges, dim=0).cpu().numpy()
                for aa, bb in edges:
                    uf.union(int(aa), int(bb))
        else:
            D = sims[s:e]
            I = idxs[s:e]
            row = np.arange(s, e, dtype=np.int64)[:, None]
            m = (D >= duplicate_thresh) & (I != row)
            if m.any():
                src = np.repeat(np.arange(s, e, dtype=np.int64)[:, None], topK, axis=1)[m]
                dst = I[m]
                a = np.minimum(src, dst)
                b = np.maximum(src, dst)
                edges = np.stack([a, b], axis=1)
                edges = np.unique(edges, axis=0)
                for aa, bb in edges:
                    uf.union(int(aa), int(bb))

        if (bi % 10 == 0) or (bi == batches):
            print(f"  [dup] batch {bi}/{batches} processed ({e - s:,} rows)")

    blackset = _keep_blackset_from_components(uf, n)
    print(f"Found {len(blackset):,} duplicate images to remove")
    return blackset

def cluster_embeddings_with_dedup(embs_norm, index, cluster_thresh, duplicate_thresh, topK, batch_size):


    def _edges_from_batch(start, D_batch, I_batch, thr, use_gpu=True):
        """
        D_batch, I_batch: (m, topK)
        thr 이상 & self 제외 → 무향(min,max) 에지 ndarray(int64, shape (E,2)) 반환
        """
        m, k = D_batch.shape
        if m == 0:
            return np.empty((0, 2), dtype=np.int64)

        if use_gpu and _TORCH_OK and torch.cuda.is_available():
            device = "cuda"
            Dt = torch.from_numpy(D_batch).to(device=device, dtype=torch.float32, non_blocking=True)
            It = torch.from_numpy(I_batch).to(device=device, dtype=torch.int64,   non_blocking=True)
            rows = torch.arange(start, start + m, device=device, dtype=torch.int64).unsqueeze(1).expand(m, k)
            mask = (Dt >= thr) & (It != rows)
            if not mask.any():
                return np.empty((0, 2), dtype=np.int64)
            src = rows[mask]
            dst = It[mask]
            a = torch.minimum(src, dst)
            b = torch.maximum(src, dst)
            edges = torch.stack((a, b), dim=1)
            edges = torch.unique(edges, dim=0).cpu().numpy()
            return edges
        else:
            rows = np.arange(start, start + m, dtype=np.int64)[:, None]
            mask = (I_batch != rows) & (D_batch >= thr)
            if not mask.any():
                return np.empty((0, 2), dtype=np.int64)
            src = np.repeat(np.arange(start, start + m, dtype=np.int64)[:, None], k, axis=1)[mask]
            dst = I_batch[mask].astype(np.int64, copy=False)
            a = np.minimum(src, dst)
            b = np.maximum(src, dst)
            edges = np.stack([a, b], axis=1)
            edges = np.unique(edges, axis=0)
            return edges

    total_start = time.time()
    n = embs_norm.shape[0]

    print("="*60)
    print("STEP 1: FAISS SEARCH (duplicates, streaming GPU/CPU)")
    print("="*60)
    dup_search_start = time.time()
    uf_dup = UnionFind(n)

    num_batches = ceil(n / batch_size)
    for batch_idx, start in enumerate(tqdm(range(0, n, batch_size), desc="FAISS search (dup)", unit="batch")):
        end = min(start + batch_size, n)
        X = np.asarray(embs_norm[start:end], dtype=np.float32, order='C')
        D, I = index.search(X, topK)  # GPU 인덱스면 실제 검색은 GPU에서 수행

        edges = _edges_from_batch(start, D, I, duplicate_thresh, use_gpu=True)
        for a, b in edges:
            uf_dup.union(int(a), int(b))

    dup_search_time = time.time() - dup_search_start

    # 연결요소 → blackset (컴포넌트당 하나만 keep: 최소 인덱스)
    comps = {}
    for i in range(n):
        r = uf_dup.find(i)
        comps.setdefault(r, []).append(i)

    blackset = set()
    for nodes in comps.values():
        if len(nodes) <= 1:
            continue
        nodes.sort()
        for v in nodes[1:]:
            blackset.add(v)

    print(f"Duplicate pass done. Duplicates removed: {len(blackset):,}")

    print("\n" + "="*60)
    print("STEP 2: FAISS SEARCH (clustering, streaming GPU/CPU)")
    print("="*60)
    cl_search_start = time.time()
    uf = UnionFind(n)

    processed = 0
    for batch_idx, start in enumerate(tqdm(range(0, n, batch_size), desc="FAISS search (cluster)", unit="batch")):
        end = min(start + batch_size, n)
        X = np.asarray(embs_norm[start:end], dtype=np.float32, order='C')
        D, I = index.search(X, topK)

        edges = _edges_from_batch(start, D, I, cluster_thresh, use_gpu=True)
        if edges.size:
            # blackset 노드 제외
            keep_mask = np.array([(a not in blackset) and (b not in blackset) for a, b in edges], dtype=bool)
            edges = edges[keep_mask]
            for a, b in edges:
                uf.union(int(a), int(b))

        processed += (end - start)
        if processed % 100000 == 0:
            print(f"  Processed {processed:,}/{n:,} non-duplicate images")

    cl_search_time = time.time() - cl_search_start

    # 그룹 수집 (blackset 제외)
    print("\n" + "="*60)
    print("STEP 3: COLLECTING GROUPS")
    print("="*60)
    collect_start = time.time()
    groups = {}
    for i in tqdm(range(n), desc="Collecting groups"):
        if i in blackset:
            continue
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    final_groups = list(groups.values())
    collect_time = time.time() - collect_start

    total_time = time.time() - total_start
    print(f"Group collection completed in {collect_time:.1f}s")
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"Total processing time: {total_time:.1f}s")
    print(f"  - FAISS dup search (GPU/CPU): {dup_search_time:.1f}s")
    print(f"  - FAISS cluster search (GPU/CPU): {cl_search_time:.1f}s")
    print(f"  - Group collection: {collect_time:.1f}s ({collect_time/total_time*100:.1f}%)")
    print(f"Original images: {n:,}")
    print(f"Duplicates removed: {len(blackset):,}")
    print(f"Final groups: {len(final_groups):,}")
    print(f"Images in groups: {sum(len(g) for g in final_groups):,}")

    return final_groups, blackset

def cluster_embeddings(embs_norm, index, thresh, topK, batch_size):
    """Original clustering function without duplicate removal"""
    n = embs_norm.shape[0]
    sims = np.zeros((n, topK), dtype=np.float32)
    idxs = np.zeros((n, topK), dtype=np.int64)

    for start in tqdm(range(0, n, batch_size), desc="FAISS search", unit="batch"):
        end = min(start + batch_size, n)
        batch_sims, batch_idxs = index.search(embs_norm[start:end], topK)
        sims[start:end] = batch_sims
        idxs[start:end] = batch_idxs

    uf = UnionFind(n)

    for i in tqdm(range(n), desc="Clustering", unit="vec"):
        for j, sim in zip(idxs[i], sims[i]):
            if sim >= thresh:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    return list(groups.values())


def save_groups(groups, filepaths, out_dir, img_size=(112, 112), save_groups=0):
    os.makedirs(out_dir, exist_ok=True)
    sorted_groups = sorted(groups, key=len, reverse=True)


    sorted_groups = sorted_groups[:save_groups]

    for idx, group in enumerate(tqdm(sorted_groups, desc="Saving groups", unit="grp")):
        fol = os.path.join(out_dir, f"group_{idx:04d}")
        if os.path.isdir(fol):
            shutil.rmtree(fol)
        os.makedirs(fol, exist_ok=True)

        saved = 0
        for i in group:
            if saved >= 10:     # ★ 클래스(그룹)당 최대 10개만 저장
                break
            src = filepaths[i]
            if os.path.isfile(src):
                img = cv2.imread(src)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    cv2.imwrite(os.path.join(fol, os.path.basename(src)), img)
                    saved += 1

def make_fr_train_labels(groups, filepaths, out_root, prefix="fr_train"):
    """
    Generate FR training label files with informative filenames.
    """
    class_names = [f"group_{i:04d}" for i in range(len(groups))]

    # Build data_list
    data_list = []
    for i, group in enumerate(groups):
        cls = class_names[i]
        for idx in group:
            path = filepaths[idx]
            data_list.append({'folder': cls, 'file_path': path})

    # Create informative filenames
    num_classes = len(class_names)
    num_data = len(data_list)

    if num_classes >= 1000:
        class_str = f"{num_classes // 1000}K_class"
    else:
        class_str = f"{num_classes}_class"

    if num_data >= 1000:
        data_str = f"{num_data // 1000}K_data"
    else:
        data_str = f"{num_data}_data"

    classes_filename = f"{prefix}_{class_str}_{data_str}_classes.npy"
    data_list_filename = f"{prefix}_{class_str}_{data_str}_data_list.npy"

    # Save with informative names
    classes = np.array(class_names)
    classes_path = os.path.join(out_root, classes_filename)
    data_list_path = os.path.join(out_root, data_list_filename)

    np.save(classes_path, classes)
    np.save(data_list_path, np.array(data_list, dtype=object))

    print(f"Saved {num_classes} classes to: {classes_filename}")
    print(f"Saved {num_data} data items to: {data_list_filename}")

    return classes_path, data_list_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster face embeddings and optionally generate FR labels.")

    parser.add_argument('--input', default='./02_input/extrac_full', type=str,
                        help='folder path to load and combine existing npy files')
    parser.add_argument('--output', type=str, default='./02_output/current')


    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--remove_duplicates', action='store_true', default=True,
                        help='Enable duplicate image removal')
    parser.add_argument('--duplicate_thresh', type=float, default=0.9,
                        help='Threshold for duplicate image removal (e.g., 0.90)')
    parser.add_argument('--cluster_thresh', type=float, default=0.7,
                        help='Threshold for clustering same person (e.g., 0.5)')
    parser.add_argument('--topK', type=int, default=100)
    parser.add_argument('--batch', type=int, default=20000000)
    parser.add_argument('--save_groups', type=int, default=50, help='Number of top groups to save')

    args = parser.parse_args()

    print(f"Starting face clustering pipeline...")
    print(f"Configuration:")
    print(f"  Cluster threshold: {args.cluster_thresh}")
    print(f"  Duplicate threshold: {args.duplicate_thresh}")
    print(f"  Remove duplicates: {args.remove_duplicates}")
    print(f"  Use GPU: {args.use_gpu}")
    print(f"  Batch size: {args.batch:,}")
    print(f"  TopK: {args.topK}")
    print("="*60)

    # Load data
    print("MODE: Loading from folder")
    embs, fpaths = load_labels_from_folder(args.input)
    print(f"Merged loaded embeddings: {len(embs):,}")
    print(f"Merged loaded filepaths: {len(fpaths):,}")



    print("\n" + "="*60)
    print("BUILDING FAISS INDEX")
    print("="*60)

    index, embs_norm = build_faiss_index(embs, use_gpu=args.use_gpu)

    # Choose clustering method
    if args.remove_duplicates:
        groups, blackset = cluster_embeddings_with_dedup(
            embs_norm, index,
            cluster_thresh=args.cluster_thresh,
            duplicate_thresh=args.duplicate_thresh,
            topK=args.topK,
            batch_size=args.batch
        )
    else:
        print("Clustering without duplicate removal...")
        groups = cluster_embeddings(embs_norm, index, thresh=args.cluster_thresh, topK=args.topK, batch_size=args.batch)
        blackset = set()

    sizes = [len(g) for g in groups]

    print("Cluster distribution:")
    distribution_lines = ["Cluster distribution:"]
    for s, cnt in zip(*np.unique(sizes, return_counts=True)):
        line = f" Size {s}: {cnt} group(s)"
        print(line)
        distribution_lines.append(line)

    if args.remove_duplicates:
        distribution_lines.append(f"Removed {len(blackset)} duplicate images")

    # Write distribution to file
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, 'clustering_result.txt')
    with open(log_path, 'w') as fw:
        for l in distribution_lines:
            fw.write(l + "\n")
    print("Cluster distribution saved to", log_path)

    if args.save_groups != 0:
        save_groups(groups, fpaths, args.output, save_groups=args.save_groups)

    # Output
    make_fr_train_labels(groups, fpaths, args.output)
    print("FR training label files generated in", args.output)
