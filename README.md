# 얼굴 임베딩 추출 & 클러스터링 파이프라인

> **구성:**


`01_extract.py`(임베딩 추출) → \
`02_cluster.py`(FAISS 기반 중복 제거 + 클러스터링 + 레이블 생성)


이 레포는 이미지 리스트를 입력으로 받아 얼굴 임베딩을 추출하고, \
추출한 임베딩으로 **FAISS** 인덱스를 구축해 **중복 제거**와 **클러스터링**을 수행한 뒤, \
학습용 `classes.npy`/`data_list.npy`를 생성합니다.

---

## 📁 디렉토리 구조 (예시)
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

# 🧰 요구 사항

conda env create -f environment.yml <env_name> \
conda activate <env_name>


### GPU 사용:

> **주의**  
> - `faiss-gpu`는 CUDA/드라이버와의 호환이 맞아야 합니다.  
---

# 🚀 빠른 시작 (End-to-End)
1) **임베딩 추출**
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
2) **클러스터링 + 중복 제거 + 레이블 생성**
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
> `--batch`는 검색 시 나눠 처리할 **행(샘플) 수**입니다. 메모리에 맞춰 조정하세요.

---

## 🧱 1단계: 임베딩 추출 — `01_extract.py` 
입력 이미지 리스트로부터 사전 훈련된 모델을 이용하여 임베딩을 생성합니다.\
ex) 01_input/5M_127M_CFSC_data_list.txt\
내부 전처리: `Resize(112×112) → ToTensor → Normalize`.\

### Argument
- `--input` *(str, default: `01_input/5M_127M_CFSC_data_list.txt`)*  \
  폴더 경로(**내부 모든 jpg/jpeg/png/bmp/webp 재귀 탐색**) \
  txt파일 내에서 1 줄당 1 경로\
- `--output` *(str, default: `./01_output/current`)*  
  결과 저장 루트\
- `--weights` *(str)*  
  `iresnet200_revised` 가중치 `.pth` 경로.
- `--batch-size` *(int, default: `33768`)*  
  배치 크기. **GPU 메모리에 맞춰 현실적인 값(예: 512~4096)**로 조정 권장.
- `--num-workers` *(int, default: `8`)*  
  DataLoader 워커 수.
- `--emb-dim` *(int, `{512, 25600}`, default: `512`)*  
- `--use-l2-norm` *(flag)*  
  임베딩 L2 정규화 후 저장.
- `--no-merge-embs` *(flag)*  
  배치별 임시 NPY로 분할 저장(디버그/초대용량 대응).
- `--split-threshold` *(int, default: `1000000`)*  
  최종 병합 저장 시 파일 분할 임계치.

### Output
- 병합 저장 시(기본):
  - `final_fpaths.npy` (리스트)  
  - `final_indices.npy` (리스트; 입력 순서 인덱스)
  - `final_embs.npy` **또는** `final_embs_norm.npy` (L2 사용 시)
- 분할 저장 시(`--no-merge-embs`):
  - `fpaths_0.npy`, `indices_0.npy`, `embs_0.npy` (또는 `embs_norm_0.npy`) …

> 모델은 자동으로 `DataParallel` 래핑되어 **모든 가용 GPU**에서 추론합니다.

---

## 🧭 2단계: 클러스터링 — `02_cluster.py`
FAISS `IndexFlatIP`(cosine과 동등; L2 정규화 후 Inner Product)로 검색 후, 
- **STEP 1** 중복 제거(연결 성분 기반)  
- **STEP 2** 동일인 클러스터링(임계치 기반 Union-Find)  
- **STEP 3** 그룹 수집을 수행합니다. 이후 학습용 라벨 파일(`classes.npy`/`data_list.npy`)을 생성합니다.

### 입력 폴더 형식
폴더 내에는 `01_extract.py`의 결과가 들어있어야 합니다.\
즉 아래 중 하나:
- `final_fpaths.npy` + `final_embs.npy`(또는 `final_embs_norm.npy`)
- 분할: `fpaths_*.npy` + `embs_*.npy`(또는 `embs_norm_*.npy`)

### Argument
- `--input` *(str, default: `./02_input/extrac_full`)*  
  위 형식의 NPY들이 들어있는 폴더.
- `--output` *(str, default: `./02_output/current`)*  
  산출물 저장 루트(레이블/미리보기 이미지/로그 등).
- `--use_gpu` *(flag, default: **True**)*  
  GPU 인덱스 사용 시도(가능하면 **모든 GPU 샤딩**).
- `--remove_duplicates` *(flag, default: **True**)*  
  중복 이미지 제거 수행.
- `--duplicate_thresh` *(float, default: `0.9`)*  
  중복 판정 유사도 임계치.
- `--cluster_thresh` *(float, default: `0.7`)*  
  동일인 클러스터링 임계치.
- `--topK` *(int, default: `100`)*  
  검색 이웃 수.
- `--batch` *(int, default: `20000000`)*  
  검색 시 한 번에 처리할 **행(샘플) 수**. 메모리에 맞춰 축소 권장(예: 수백만/수십만 단위).
- `--save_groups` *(int, default: `50`)*  
  상위 N개 그룹을 미리보기 이미지로 저장(그룹당 최대 10장, 112×112 리사이즈).

### Output
- **FAISS 검색·클러스터 통계 로그**: 콘솔 + 파일(`clustering_result.txt`)  
- **그룹 미리보기 이미지**: `output/` 하위 `group_0000/…` 형식 (요청된 경우)  
- **학습용 라벨 파일** *(자동 생성)*: `output/` 하위
  - `fr_train_{X_class|XK_class}_{Y_data|YK_data}_classes.npy`
  - `fr_train_{...}_data_list.npy`  
  예: `fr_train_12K_class_3K_data_classes.npy`


---

## 📊 파라미터 가이드
- **정규화 & 메트릭**: `02_cluster.py`에서 임베딩을 **L2 정규화** 후 `IndexFlatIP` 사용 → 코사인 유사도와 동일 동작.  
  (이미 `01_extract.py`에서 `--use-l2-norm`으로 저장했어도, 02 단계에서 한 번 더 정규화합니다.)
- **임계치**: 데이터 품질/난이도에 따라 `duplicate_thresh`(0.9±), `cluster_thresh`(0.6~0.8)를 조정하세요.  
- **GPU 인덱스 확인**: 로그의 `Sharded index across N GPU(s)` 문구로 GPU 사용 여부를 확인합니다.

---

## 🛠️ 문제 해결 (Troubleshooting)
- **`CUBLAS_STATUS_EXECUTION_FAILED`**  
  CUDA/드라이버 불일치 또는 메모리 부족 가능성. 배치 축소, FP16 해제, 버전 정렬을 시도하세요.
- **`AxisError: axis 1 is out of bounds for array of dimension 1`**  
  입력 임베딩이 2D가 아닌 경우. 파일 로드 후 `arr.shape` 확인, `np.asarray(arr, dtype=np.float32)`로 강제 변환.
- **GPU인데 속도 차이 미미**  
  실제로 **GPU 인덱스**가 생성되었는지, `topK`/`nprobe`(IVF 사용 시) 등의 검색 파라미터가 병목인지 확인.
- **메모리 이슈**  
  `--batch`(02단계)와 `--batch-size`(01단계)를 **감소**하여 스트리밍 처리하세요.

---

## 🔁 재현성 & 로깅 체크리스트
- [ ] `01_extract.py` 저장물: `final_*.npy` 또는 분할 `*_i.npy` 페어 존재  
- [ ] `02_cluster.py` 입력 폴더에 임베딩/경로 파일이 모두 존재  
- [ ] `--remove_duplicates` ON 상태에서 중복 제거 수 확인  
- [ ] `classes.npy` ↔ `data_list.npy` 페어링 정합 확인 (인덱스 기준)

---

## 📄 라이선스 / 📬 문의
- SECERN AI 

