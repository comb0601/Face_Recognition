import os
import shutil
from pathlib import Path
import time
from typing import Literal
from math import ceil
from functools import reduce

import torch
import cv2
import numpy as np
import cupy as cp
from tqdm import tqdm
import faiss
import torchvision.transforms as tvf

from ai_model import iresnet200_revised
from utils import (
    l2_normalize, merge_groups, merge_groups_fast, all_gather, synchronize_processes, 
    get_latency, make_folder, LatencyChecker, l2_norm_tensor, cosine_similarity, cosine_similarity_tensor
)


ltc = LatencyChecker()

transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Resize((112,112)),
            tvf.Normalize(0.5, 0.5)
        ])

def _get_merged_class_name_from_path(x):
    if 'linkedin' in x:
        return _get_linkedin_class_name_from_path(x)
    else:
        return _get_cfsc_class_name_from_path(x)

def _get_linkedin_class_name_from_path(x):
    return x.split('/')[-1][:-4] # "40d2710c8570c4f2_face_5"

def _get_cfsc_class_name_from_path(x):
    return '/'.join(x.split('/')[-3:-1]) # "1_CFSC_420k_jpg/m.01fkr6"

def _concat_npys(npy_path_list, concat_dim=0, name=""):
    npy_list = [
        np.load(npy_path) for npy_path in tqdm(npy_path_list, desc=f'loading {name} npys..')
    ]

    if not npy_list:
        return np.array([])  # 빈 목록 처리

    first_npy_shape = npy_list[0].shape
    num_dims = len(first_npy_shape)

    total_size = sum(npy.shape[concat_dim] for npy in npy_list)

    final_shape = list(first_npy_shape)
    final_shape[concat_dim] = total_size
    final_shape = tuple(final_shape)

    npys = np.zeros(final_shape, dtype=npy_list[0].dtype)

    temp_start_point = 0
    for npy in npy_list:
        b_size = npy.shape[concat_dim]
        slices = [slice(None)] * num_dims
        slices[concat_dim] = slice(temp_start_point, temp_start_point + b_size)
        npys[tuple(slices)] = npy
        temp_start_point += b_size

    return npys

@get_latency
def sort_paired_matrices(A, B):
    descending_indices = np.argsort(A, axis=1)[:, ::-1]
    sorted_A = np.take_along_axis(A, descending_indices, axis=1)
    sorted_B = np.take_along_axis(B, descending_indices, axis=1)

    return sorted_A, sorted_B

class FR_Embedding_Extractor:
    def __init__(self, ):
        self.fr_encoder = iresnet200_revised()
        self.fr_encoder.load_state_dict(
            torch.load('/purestorage/project/hkl/hkl_slurm/codes/ArtfaceStudio_ML/train/t2i_with_extension/weights/AVG_pruned_weight.pth'))

    def _get_tensor_from_fpath_list(self, fpath_list: list[str]):
        rgb_face_norm_tensor_list = []
        for fpath in fpath_list:
            bgr_img = cv2.imread(fpath)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_face_norm_tensor = transform(rgb_img)
            rgb_face_norm_tensor_list.append(rgb_face_norm_tensor.unsqueeze(0))
        
        rgb_face_norm_tensor = torch.cat(rgb_face_norm_tensor_list, dim=0)
        return rgb_face_norm_tensor

    def run(self, source_dataloader, save_path='./outs/fr_emb', use_l2_norm=False, emb_dim=512, merge_embs=True):
        self.source_dataloader = source_dataloader
        self.save_path = save_path

        # 진행바 + 폴더 생성
        self.source_dataloader = tqdm(self.source_dataloader)
        make_folder(self.save_path, remove_if_exist=True)

        # DataParallel로 모델 여러 GPU에 배치
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fr_encoder = torch.nn.DataParallel(self.fr_encoder).cuda().eval()

        embs_name = 'embs_norm' if use_l2_norm else 'embs'
        emb_list = []
        fpath_list = []

        # 배치별 추론
        for bidx, batch in enumerate(self.source_dataloader):
            rgb_face_norm_tensor = batch['rgb_face_norm_tensor']
            jpg_path_list = batch['jpg_path']

            with torch.no_grad():
                if emb_dim == 512:
                    embs = self.fr_encoder(rgb_face_norm_tensor.to(self.device, non_blocking=True))
                    if use_l2_norm:
                        embs_norm = torch.norm(embs, p=2, dim=1, keepdim=True)
                        embs = embs / (embs_norm + 1e-12)
                elif emb_dim == 25600:
                    mid_embs, final_embs = self.fr_encoder(rgb_face_norm_tensor.cuda(), do_mid_val=True)
                    final_embs = l2_norm_tensor(final_embs) if use_l2_norm else final_embs
                    embs = torch.cat([mid_embs, final_embs], dim=1)
                else:
                    raise ValueError(f'emb_dim: {emb_dim} is not supported')

            rgb_face_norm_tensor.detach().cpu()
            embs = embs.detach().cpu()

            if not merge_embs:
                np.save(f'{self.save_path}/temp_{bidx}_{embs_name}.npy', embs.numpy())
            else:
                emb_list.append(embs)
                fpath_list += jpg_path_list

            del embs, rgb_face_norm_tensor
            torch.cuda.empty_cache()

        # 병합 저장
        if merge_embs:
            embs = torch.cat(emb_list, dim=0).numpy()
            num_emb = embs.shape[0]
            THRESHOLD_TO_SPLIT = 100 * 10000

            if num_emb > THRESHOLD_TO_SPLIT:
                num_to_split = ceil(num_emb / THRESHOLD_TO_SPLIT)
                for i in range(num_to_split):
                    np.save(f'{self.save_path}/fpaths_{i}.npy',
                            fpath_list[i*THRESHOLD_TO_SPLIT:(i+1)*THRESHOLD_TO_SPLIT])
                    np.save(f'{self.save_path}/{embs_name}_{i}.npy',
                            embs[i*THRESHOLD_TO_SPLIT:(i+1)*THRESHOLD_TO_SPLIT])
            else:
                np.save(f'{self.save_path}/final_fpaths.npy', fpath_list)
                np.save(f'{self.save_path}/final_{embs_name}.npy', embs)

    
    def run_variface(self, out_folder_path='/purestorage/AILAB/AI_1/dataset/hkl_temp/variface'):
        ''' sorted by folder name
        [
        {'folder': '1_CFSC_420k_jpg/g.112ydl6mg', 'file_path': '...'},
        {'folder': '1_CFSC_420k_jpg/g.112ydl6mg', 'file_path': '...'},
        ...
        {'folder': '1_CFSC_420k_jpg/g.112yf7h6d', 'file_path': '...'},
        {'folder': '1_CFSC_420k_jpg/g.112yf7h6d', 'file_path': '...'},
        ...
        ]
        '''
        data_list = np.load(
            '/purestorage/datasets/face_recognition/000_classes_data_list/cfsc/cfsc_50u_95d_data_list_ipi-20u.npy', 
            allow_pickle=True)
        
        file_path_list_map = {}
        for data in data_list:
            folder_name = data['folder']
            file_path = data['file_path']
            if folder_name not in file_path_list_map:
                file_path_list_map[folder_name] = []
            file_path_list_map[folder_name].append(file_path)
        
        make_folder(f'{out_folder_path}/embs', remove_if_exist=True)
        
        f = open(f'{out_folder_path}/data_list.txt', 'w')
        for folder_name, file_path_list in tqdm(list(file_path_list_map.items()), desc='processing...'):
            new_folder_name = folder_name.replace('/', '_')
            rgb_face_norm_tensor = self._get_tensor_from_fpath_list(file_path_list)

            with torch.no_grad():
                mid_embs, final_embs = self.fr_encoder(rgb_face_norm_tensor.cuda(), do_mid_val=True)
                avg_final_embs = torch.mean(final_embs, keepdim=True, dim=0)

                divergence_scores = cosine_similarity_tensor(avg_final_embs, final_embs)

                fr_emb_for_condition = torch.cat([mid_embs, l2_norm_tensor(final_embs)], dim=1)
                avg_fr_emb_for_condition = torch.mean(fr_emb_for_condition, dim=0)

                divergence_scores = divergence_scores.detach().cpu()
                avg_fr_emb_for_condition = avg_fr_emb_for_condition.detach().cpu()
                del mid_embs, final_embs, avg_final_embs
                torch.cuda.empty_cache()

            for i in range(len(file_path_list)):
                f.write(f'{new_folder_name},{file_path_list[i]},{divergence_scores[i].item()}\n')
            np.save(f'{out_folder_path}/embs/{new_folder_name}.npy', avg_fr_emb_for_condition.numpy())

        f.close()
        
class FaceClusterProcessor:
    def __init__(self, 
                 emb_npy_path_list=['outs/cubox_fas_embs.npy'], 
                 fpath_npy_path_list=['outs/cubox_fas_fnames.npy'], 
                 use_gpu=True,
                 do_l2_norm=False,
                 same_person_cos_threshold = 0.5,
                 same_image_cos_threshold = None,
                 ):
        self.emb_dimension = 512
        self.use_gpu = use_gpu
        self.same_person_cos_threshold = same_person_cos_threshold
        self.same_image_cos_threshold = same_image_cos_threshold

        self.indice_whiteset = set([])
        self.indice_blackset = set([])

        embs, fpaths = self._concat_inputs(emb_npy_path_list, fpath_npy_path_list, use_l2_norm=do_l2_norm)
        self.embs = embs
        self.fpaths = fpaths

        self.num_emb = self.fpaths.shape[0] if self.embs is None else self.embs.shape[0]

    @get_latency
    def _concat_inputs(self, emb_npy_path_list, fpath_npy_path_list, use_l2_norm=True):
        print('loading npy files...')
        embs = None
        if emb_npy_path_list is not None:
            emb_npys = [torch.Tensor(np.load(emb_npy_path)) for emb_npy_path in tqdm(emb_npy_path_list, desc='loading emb..')]
            embs_tensor = torch.cat(emb_npys, dim=0)
            embs = embs_tensor.numpy()

            if use_l2_norm:
                embs = l2_normalize(embs, axis=1)

        fpaths = None
        if fpath_npy_path_list is not None:
            fpaths = _concat_npys(npy_path_list=fpath_npy_path_list, name='fpath')

        return embs, fpaths

    @get_latency
    def _make_index(self, emb_start_index=None, size=None):
        print('making index...')
        if 1:
            cpu_index = faiss.IndexFlatIP(self.emb_dimension)
        elif 0: # experimental: work, but not that fast.
            HNSW_N = 4
            cpu_index = faiss.IndexHNSWFlat(self.emb_dimension, HNSW_N, faiss.METRIC_INNER_PRODUCT) # IndexHNSWFlat(int d, int M, MetricType metric)
            cpu_index.hnsw.efSearch = 16  # 검색 정확도 조정
            cpu_index.hnsw.efConstruction = 100  # 그래프 구성 정확도 조정

        if emb_start_index is not None and size is not None:
            embs_to_add = self.embs[emb_start_index:emb_start_index+size]
        else:
            embs_to_add = self.embs

        if self.use_gpu:
            ngpus = faiss.get_num_gpus()

            print("number of GPUs:", ngpus)

            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            # co.useFloat16 = True
            # co = None

            gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
                cpu_index, co=co
            )
            gpu_index.add(embs_to_add)
            final_index = gpu_index
        else:
            cpu_index.add(embs_to_add)
            final_index = cpu_index

        return final_index

    def _get_too_same_img_indice_blackset(self, cosineSimTopKList, indexTopKList, emb_start_index=0):
        if self.same_image_cos_threshold is None:
            return

        for search_index in tqdm(list(range(self.num_emb)), desc="make blacklist.."):
            cosineSimTopK = cosineSimTopKList[search_index]
            indexTopK = indexTopKList[search_index]
 
            index_first = indexTopK[0]
            real_index_first = index_first + emb_start_index
            real_index_first = real_index_first.item()
            is_self = (real_index_first == search_index)
            if is_self:
                if real_index_first not in self.indice_blackset:
                    self.indice_whiteset.add(real_index_first)

                target_cosineSimTopK = cosineSimTopK[1:]
                target_indexTopK = indexTopK[1:]
            else:
                target_cosineSimTopK = cosineSimTopK
                target_indexTopK = indexTopK

            # 유사도 임계값 초과하는 index 추출 (result_idx > 0)
            too_same_mask = target_cosineSimTopK > self.same_image_cos_threshold # result_idx 1부터 비교
            too_same_indices = target_indexTopK[too_same_mask] # result_idx 1부터 해당 index 추출

            # blackset에 추가 (whiteset에 없는 index만)
            for index in too_same_indices:
                real_index = index + emb_start_index
                real_index = real_index.item()
                if real_index in self.indice_whiteset: 
                    if search_index not in self.indice_whiteset: 
                        self.indice_blackset.add(search_index) # 검색 결과는 이미 whitelist -> 검색 쿼리에 대해서
                else: 
                    self.indice_blackset.add(real_index) # 검색 결과에 대해서
    
    @get_latency
    def _get_too_same_img_indice_blackset_v2(self, cosineSimTopKList, indexTopKList, emb_start_index=0):
        if self.same_image_cos_threshold is None:
            return
        
        indexTopKList = indexTopKList + emb_start_index
        
        too_same_mask = cosineSimTopKList > self.same_image_cos_threshold
        row_indices, col_indices = np.where(too_same_mask)

        too_same_pair_list = []
        for i in range(len(row_indices)):
            row_idx = row_indices[i]
            col_idx = col_indices[i]
            too_same_pair_list.append([row_idx, indexTopKList[row_idx, col_idx]])

        merged_pair_list = merge_groups_fast(too_same_pair_list)

        for pair in merged_pair_list:
            if len(pair) > 1:
                self.indice_blackset.update(pair[1:])

    def _make_samples(self, out_root_path, same_person_indice_list):
        tmp_root_path = f'{out_root_path}/samples'
        if os.path.isdir(tmp_root_path):
            shutil.rmtree(tmp_root_path)

        cnt = 0
        for sidx, same_person_indice in enumerate(same_person_indice_list):
            # if len(same_person_indice) < 2:
            #     continue
            
            cnt += 1
            
            tmp_fol_name = f'{tmp_root_path}/{str(sidx).zfill(4)}'
            make_folder(tmp_fol_name, remove_if_exist=True)

            random_color = np.random.randint(0, 256, 3).tolist()
            for idx in same_person_indice:
                jpg_path = self.fpaths[idx]
                jpg_name = jpg_path.split('/')[-1]

                # shutil.copy(
                #     jpg_path, 
                #     f'{tmp_fol_name}/{jpg_name}')
            
                img = cv2.imread(jpg_path)
                img = cv2.resize(img, (112,112))
                img[100:112, :, :] = random_color
                cv2.imwrite(f'{tmp_fol_name}/{jpg_name}', img)
            


    def _make_label_for_fr_train(self, out_root_path, merged_same_person_indice_list):
        class_name_list = [_get_merged_class_name_from_path(fpath) for fpath in tqdm(self.fpaths, desc="making class name list")]
        class_map = {}

        for indice in tqdm(merged_same_person_indice_list, desc="making class map"):
            target_class_idx = indice[0]
            target_class_name = class_name_list[target_class_idx]
            class_map[target_class_name] = target_class_name

            if len(indice) > 1:
                for class_idx in indice[1:]:
                    class_name = class_name_list[class_idx]
                    class_map[class_name] = target_class_name

        
        classes = list(class_map.values())
        classes = np.array(classes)
        classes = np.unique(classes)
        classes = list(classes)
        classes.sort()

        data_list = []

        ''' Hard-coded CONFIG '''
        CFSC_ALL_JPG_PATH_FILE = '/purestorage/datasets/face_recognition/CFSC_data/all_jpg_paths.txt'
        #LINKEDIN_VERSION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        LINKEDIN_VERSION_LIST = [11]
        ''' CFSC '''
        f = open(CFSC_ALL_JPG_PATH_FILE, 'r')
        cnt = 0
        miss_cnt = 0
        err_cnt = 0
        while True:
            l = f.readline()
            if not l: break
            cnt += 1
            # if cnt % 1000000 == 0:
            #     print(f'cnt: {cnt:,}')

            jpg_path = l.strip()
            class_name = _get_cfsc_class_name_from_path(jpg_path)
            try:
                if class_name in class_map:
                    target_class_name = class_map[class_name]
                    data = {'folder': target_class_name, 'file_path': jpg_path}
                    data_list.append(data)
                else:
                    miss_cnt += 1
            except Exception as e:
                err_cnt += 1
                # print(e)

        f.close()
        print(f'[CFSC] err_cnt: {err_cnt}, miss_cnt: {miss_cnt}, left_cnt: {cnt - miss_cnt}')

        ''' linkedin '''
        cnt = 0
        miss_cnt = 0
        err_cnt = 0
        for version_idx in LINKEDIN_VERSION_LIST:
            f = open(f'/purestorage/AILAB/AI_1/dataset/linkedin_faces/v{version_idx}_jpg_paths.txt', 'r')
            while True:
                l = f.readline()
                if not l: break
                cnt += 1
                # if cnt % 300000 == 0:
                #     print(f'cnt: {cnt:,}')

                jpg_path = l.strip()
                class_name = _get_linkedin_class_name_from_path(jpg_path)
                try:
                    if class_name in class_map:
                        target_class_name = class_map[class_name]
                        data = {'folder': target_class_name, 'file_path': jpg_path}
                        data_list.append(data)
                    else:
                        miss_cnt += 1
                except Exception as e:
                    err_cnt += 1
                    # print(e)

            f.close()

        print(f'[linked] err_cnt: {err_cnt}, miss_cnt: {miss_cnt}, left_cnt: {cnt - miss_cnt}')

        num_classes = len(classes)
        num_data = len(data_list)
        print('# of classes: ', num_classes)
        print('# of data: ', num_data)

        np.save(f'{out_root_path}/fr-train_{num_classes//1000}k_{num_data//1000000}M-classes.npy', classes)
        np.save(f'{out_root_path}/fr-train_{num_classes//1000}k_{num_data//1000000}M-data_list.npy', data_list)

    @get_latency
    def _merge_and_save_results(self, out_root_path, same_person_indice_list):
        print('merge group...')
        merged_same_person_indice_list = merge_groups_fast(same_person_indice_list)

        f = open(f'{out_root_path}/indice_group.txt', 'w')
        for indice in merged_same_person_indice_list:
            f.write(f'{indice}\n'.replace('[', '').replace(']', ''))
        f.close()
        
        return merged_same_person_indice_list

    def _search_and_get_same_person_indice_list(
            self, 
            faiss_index, 
            topK, 
            emb_start_index=0,
            cosineSimTopKList=None,
            indexTopKList=None,
        ):
        same_person_indice_list = []

        print('searching...')
        '''
        cosineSimTopKList: [[0.99999994 0.2309655  0.2309655 ...
        indexTopKList: [[    0  7572  7981  4536 11758 ...
        '''
        if cosineSimTopKList is None or indexTopKList is None:
            # Process in batches to avoid OOM
            st_search = time.perf_counter()
            
            # Define batch size to avoid OOM (adjust based on available memory)
            batch_size = 10000  # Adjust this value based on your system's memory capacity
            num_batches = ceil(self.num_emb / batch_size)
            
            # Initialize result arrays
            cosineSimTopKList = np.zeros((self.num_emb, topK), dtype=np.float32)
            indexTopKList = np.zeros((self.num_emb, topK), dtype=np.int32)
            
            for batch_idx in tqdm(range(num_batches), desc='Batch searching'):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_emb)
                
                # Search for current batch
                batch_cosineSimTopK, batch_indexTopK = faiss_index.search(
                    self.embs[start_idx:end_idx], topK
                )
                
                # Store results for this batch
                cosineSimTopKList[start_idx:end_idx] = batch_cosineSimTopK
                indexTopKList[start_idx:end_idx] = batch_indexTopK
            
            et_search = time.perf_counter()
            print(f'search latency: r{et_search - st_search} secs')


        # self._get_too_same_img_indice_blackset_v2(cosineSimTopKList, indexTopKList, emb_start_index) # too same images to exclude
        self._get_too_same_img_indice_blackset(cosineSimTopKList, indexTopKList, emb_start_index) # too same images to exclude
        indice_blackset_list = list(self.indice_blackset)

        cosineSimTopKList_np = np.array(cosineSimTopKList) # (num_emb, topK)
        indexTopKList_np = np.array(indexTopKList)         # (num_emb, topK)
        real_indexTopKList_np = indexTopKList_np + emb_start_index

        # 1. 블랙리스트 마스크 생성 (NumPy Broadcasting 활용)
        mask_not_in_blacklist = ~np.isin(real_indexTopKList_np, indice_blackset_list) # (num_emb, topK) - Boolean Mask

        # 2. 유사도 기준 마스크 생성 (NumPy Broadcasting 활용)
        mask_same_person = cosineSimTopKList_np > self.same_person_cos_threshold # (num_emb, topK) - Boolean Mask

        # 3. 두 마스크 AND 연산 (요소별 AND)
        final_mask = mask_not_in_blacklist & mask_same_person # (num_emb, topK) - 최종 Boolean Mask

        # 4. 최종 마스크를 index 에 적용하여 필터링된 index 추출
        for i in tqdm(list(range(self.num_emb)), desc='find similar faces..'): # num_emb 만큼 순회 (각 검색 대상에 대해)
            filtered_indexTopK = real_indexTopKList_np[i][final_mask[i]] # (filtered_count) - 각 검색 대상별 필터링된 index

            if len(filtered_indexTopK) > 0:
                filtered_indexTopK_adjusted = filtered_indexTopK # + emb_start_index
                filtered_indexTopK_adjusted = filtered_indexTopK_adjusted.tolist()
                if i not in self.indice_blackset:
                    filtered_indexTopK_adjusted.append(i) 
                same_person_indice_list.append(filtered_indexTopK_adjusted)

        return same_person_indice_list

    def run(
        self, 
        topK=100, 
        out_root_path='outs/250219', 
        make_sample_folder=False,
        threshold_num_emb_to_split = 1000 * 10000, # GPU: 10M = 52GB, 1M = 8GB
        make_fr_label_for_cfsc_and_linkedin=False
    ):
        make_folder(out_root_path, remove_if_exist=False)

        if self.num_emb > threshold_num_emb_to_split:
            num_to_index = ceil(self.num_emb / threshold_num_emb_to_split)

            final_same_person_indice_list = []
            for index_number in range(num_to_index):
                emb_start_index = index_number*threshold_num_emb_to_split
                final_index = self._make_index(
                    emb_start_index=emb_start_index, 
                    size=threshold_num_emb_to_split)
            
                same_person_indice_list = self._search_and_get_same_person_indice_list(
                    final_index, topK, emb_start_index=emb_start_index)
                final_same_person_indice_list += same_person_indice_list

        else:
            final_index = self._make_index()
            final_same_person_indice_list = self._search_and_get_same_person_indice_list(
                final_index, topK, emb_start_index=0)

        merged_same_person_indice_list = self._merge_and_save_results(out_root_path, final_same_person_indice_list)

        if make_sample_folder:
            self._make_samples(out_root_path, merged_same_person_indice_list)

        if make_fr_label_for_cfsc_and_linkedin:
            self._make_label_for_fr_train(out_root_path, merged_same_person_indice_list)

    def run_index_search(
        self, 
        topK=100, 
        out_root_path='outs/250219', 
        threshold_num_emb_to_split = 1000 * 10000, # GPU: 10M = 52GB, 1M = 8GB
        node_rank=0,
        nnodes=1,
    ):
        quota_per_node = self.num_emb // nnodes + self.num_emb % nnodes

        if quota_per_node > threshold_num_emb_to_split:
            quota = threshold_num_emb_to_split
        else:
            quota = quota_per_node
  
        num_to_index = ceil(self.num_emb / quota) # ex) 30500 / 1000 -> 31

        num_index_quota_per_node = num_to_index // nnodes # ex) 31 // 3 -> 10
        num_index_left = num_to_index % nnodes # ex) 31 % 2 -> 1

        for index_number in range(num_index_quota_per_node): # ex) 0~9
            job_index = index_number + num_index_quota_per_node * node_rank
            emb_start_index = job_index * quota # rank 0: 0~9 * 1000, rank 1: 10~19 * 1000, rank 2: 20~29 * 1000
            final_index = self._make_index(
                emb_start_index=emb_start_index, 
                size=quota,
            )

            ltc.s()
            cosineSimTopKList, indexTopKList = final_index.search(self.embs, topK)
            ltc.e(f'[{node_rank}]search-{job_index}')

            np.save(f'{out_root_path}/search/cosine_{job_index}.npy', cosineSimTopKList)
            np.save(f'{out_root_path}/search/index_{job_index}.npy', np.array(indexTopKList) + emb_start_index)
        
        if node_rank == nnodes-1: # left to last rank
            for index_number in range(num_index_left):
                job_index = index_number + num_index_quota_per_node * (node_rank + 1)
                emb_start_index = job_index * quota # rank 2: 30 * 1000
                final_index = self._make_index(
                    emb_start_index=emb_start_index, 
                    size=quota,
                )

                ltc.s()
                cosineSimTopKList, indexTopKList = final_index.search(self.embs, topK)
                ltc.e(f'[{node_rank}]search-{job_index}')

                np.save(f'{out_root_path}/search/cosine_{job_index}.npy', cosineSimTopKList)
                np.save(f'{out_root_path}/search/index_{job_index}.npy', np.array(indexTopKList) + emb_start_index)

    def run_get_same_person_group_from_search_result(
        self,
        out_root_path,
        search_cos_npy_path_list=[],
        search_idx_npy_path_list=[],
        make_sample_folder=False,
        make_fr_label_for_cfsc_and_linkedin=False,
    ):
        assert len(search_cos_npy_path_list) == len(search_idx_npy_path_list), 'size mismatch between cos and idx'

        do_concat_and_search = False

        if do_concat_and_search:
            final_cos_npy = _concat_npys(search_cos_npy_path_list, concat_dim=1, name='cos')
            final_idx_npy = _concat_npys(search_idx_npy_path_list, concat_dim=1, name='idx')
            final_cos_npy, final_idx_npy = sort_paired_matrices(final_cos_npy, final_idx_npy) # 100 min when 40M x 600 datas

            final_same_person_indice_list = self._search_and_get_same_person_indice_list( # 90 min when 40M x 600 blacklist..
                faiss_index=None, topK=None, emb_start_index=0,
                cosineSimTopKList=final_cos_npy,
                indexTopKList=final_idx_npy
            )
        else:
            final_same_person_indice_list = []

            for idx in range(len(search_cos_npy_path_list)):
                ltc.s()
                temp_cos_npy = np.load(search_cos_npy_path_list[idx])
                ltc.e('load_cos')
                ltc.s()
                temp_idx_npy = np.load(search_idx_npy_path_list[idx])
                ltc.e('load_idx')
                temp_same_person_indice_list = self._search_and_get_same_person_indice_list( # 13 min + 1 min when 40M x 100 blacklist
                    faiss_index=None, topK=None, emb_start_index=0,
                    cosineSimTopKList=temp_cos_npy,
                    indexTopKList=temp_idx_npy
                )
                final_same_person_indice_list += temp_same_person_indice_list

        merged_same_person_indice_list = self._merge_and_save_results(out_root_path, final_same_person_indice_list)

        if make_sample_folder:
            self._make_samples(out_root_path, merged_same_person_indice_list)

        if make_fr_label_for_cfsc_and_linkedin:
            self._make_label_for_fr_train(out_root_path, merged_same_person_indice_list)
        
    def run_make_label_for_fr_train(self, out_root_path, indice_group_txt_path):
        '''
        Warning: self.fpaths order should be matched with indice group data.
        '''
        make_folder(out_root_path, remove_if_exist=False)

        f = open(f'{indice_group_txt_path}', 'r')
        indice_group = []
        while True:
            l = f.readline()
            if not l: break
            indice = l.strip().split(',')
            indice_group.append([int(idx) for idx in indice])
        f.close()

        self._make_label_for_fr_train(
            out_root_path=out_root_path, 
            merged_same_person_indice_list=indice_group)



def temp_triplet_fas():
    import json
    f = open('outs/250115/cubox_fas_indice_group.txt', 'r')

    indice_group = []
    while True:
        l = f.readline()
        if not l: break
        indice = l[:-1].split(',')
        indice_group.append([int(idx) for idx in indice])

    f.close()

    root_path = '/purestorage/datasets/face_anti_spoofing/CUBOX_RGB_FAS_1k_people/final_data/data_jpg_and_json'
    fnames = np.load('outs/250115/cubox_fas_fnames.npy')

    for face_id, indice in tqdm(list(enumerate(indice_group))):
        if len(indice) < 400 or len(indice) > 1000:
            continue

        for idx in indice:
            jpg_fname = fnames[idx] # display_fake_data_6584_pid2_cnt15.jpg
            json_fname = jpg_fname.split('.')[0] + '.json'

            unique_scene_str = jpg_fname.split('_pid')[0]
            frame_cnt_str = jpg_fname.split('cnt')[1].split('.')[0]

            f = open(f'{root_path}/{json_fname}', 'r')
            anno = json.load(f)
            f.close()

            anno['img_path'] = f'{root_path}/{jpg_fname}'
            anno['frame_cnt'] = int(frame_cnt_str)
            anno['scene_str'] = unique_scene_str
            anno['face_id'] = face_id

            f = open(f'/purestorage/datasets/face_anti_spoofing/CUBOX_RGB_FAS_ALL/label_for_triplet_fas2/{json_fname}', 'w')
            json.dump(anno, f)
            f.close()

def temp_make_annotation_for_gen_data_for_fr(jpg_path_list, name, limit_imgs_per_class=-1, additive_name=''):
    last_fol_name = name
    
    class_to_pathlist = {}
    classes = set([])
    data_list = []
    for jpg_path in tqdm(jpg_path_list):
        parents = jpg_path.parents

        if parents[1].name == last_fol_name:
            class_name = parents[0].name # folder name
        else:
            class_name = '/'.join([parents[1].name, parents[0].name]) # CFSC folder name (e.g. CFSC/m.043rt)
        
        data = {'folder': class_name, 'file_path': str(jpg_path)}
        classes.add(class_name)
        data_list.append(data)

        if class_name in class_to_pathlist:
            class_to_pathlist[class_name].append(str(jpg_path))
        else:
            class_to_pathlist[class_name] = [str(jpg_path)]

    if limit_imgs_per_class > 0:
        data_list = []

        for class_name, jpg_path_list in class_to_pathlist.items():
            origin_img_path = None
            left_img_path_list = []

            for jpg_path in jpg_path_list:
                if 'origin.jpg' in jpg_path:
                    origin_img_path = jpg_path
                else:
                    left_img_path_list.append(jpg_path)
            
            assert origin_img_path is not None, "origin_img_path is None!"
            
            data_list.append({'folder': class_name, 'file_path': origin_img_path})

            num_to_push = min(len(left_img_path_list), limit_imgs_per_class-1)
            for idx in range(num_to_push):
                data_list.append({'folder': class_name, 'file_path': left_img_path_list[idx]})

    
        
    fol = '.'
    # fol = '/purestorage/datasets/face_recognition/000_classes_data_list/exp'

    print('# of class: ', len(classes))
    print('# of data: ', len(data_list))
    num_classes_k_str = f'{len(classes) // 1000}k'
    num_data_k_str = f'{len(data_list) // 1000}k'
    data_id = f'{num_classes_k_str}-c_{num_data_k_str}-d'
    np.save(f'{fol}/{name}_{additive_name}_{data_id}_classes.npy', list(classes))
    np.save(f'{fol}/{name}_{additive_name}_{data_id}_data_list.npy', data_list) 

def temp_l2_norm_process():
    # embs = np.load(f'outs/cfsc_all/final_embs.npy')
    # embs = l2_normalize(embs, axis=1)
    # np.save(f'outs/cfsc_all/final_embs_norm.npy', embs)

    for i in tqdm(list(range(0, 34))):
        embs = np.load(f'outs/linkedin_v5/s_{i}.npy')
        embs = l2_normalize(embs, axis=1)
        np.save(f'outs/linkedin_v5/embs_norm_{i}.npy', embs)

def temp_extract_only_one_img_per_class():
    # version = 'v11'
    # file_num = 26
    # emb_npy_path_list = [f'outs/linkedin_{version}/fre_data/embs_norm_{i}.npy' for i in range(0, file_num)]
    # fpath_npy_path_list = [f'outs/linkedin_{version}/fre_data/fpaths_{i}.npy' for i in range(0, file_num)]
    # indice_group_txt_path = f'outs/linkedin_{version}/indice_group.txt'

    emb_npy_path_list = [f'outs/linkedin_v{v}_refined/embs_norm.npy' for v in range(1, 12)]
    fpath_npy_path_list = [f'outs/linkedin_v{v}_refined/fpaths.npy' for v in range(1, 12)]
    indice_group_txt_path = f'outs/merged_lin-r-v1-v11/indice_group.txt'
    
    # fol_name = f'outs/linkedin_{version}_refined'
    fol_name = f'outs/merged_lin-r-v1-v11_refined'
    make_folder(fol_name)

    indice = []

    f = open(f'{indice_group_txt_path}', 'r')
    while True:
        l = f.readline()
        if not l: break
        idx = int(l.split(',')[0].strip())
        indice.append(idx)
    f.close()

    emb_npys = [torch.Tensor(np.load(emb_npy_path)) for emb_npy_path in tqdm(emb_npy_path_list, desc='loading emb..')]
    embs_tensor = torch.cat(emb_npys, dim=0)
    embs = embs_tensor.numpy()
    new_embs = embs[indice]
    ltc.s()
    np.save(f'{fol_name}/embs_norm.npy', new_embs)
    ltc.e('save embs')

    fpaths = _concat_npys(npy_path_list=fpath_npy_path_list, name='fpath')
    new_fpaths = fpaths[indice]
    ltc.s()
    np.save(f'{fol_name}/fpaths.npy', new_fpaths)
    ltc.e('save fpaths')


def temp_make_variface_annotation():
    def _process_fol(key, datas):
        # Process the embeddings for this key
        key_condition_embs = np.array([data['emb'] for data in datas])
        fpaths = [data['fpath'] for data in datas]

        try:
            _, key_final_embs = key_condition_embs[:, :-512], key_condition_embs[:, -512:]
        except:
            print(key, datas)
            import sys
            sys.exit()


        avg_key_condition_emb = np.mean(key_condition_embs, axis=0, keepdims=True)
        
        avg_key_final_emb = np.mean(key_final_embs, axis=0, keepdims=True)
        avg_key_final_emb_tensor = torch.from_numpy(avg_key_final_emb)
        key_final_embs_tensor = torch.from_numpy(key_final_embs)

        divergence_scores = cosine_similarity_tensor(avg_key_final_emb_tensor, key_final_embs_tensor)

        return {
            'divergence_scores': divergence_scores.numpy(), 
            'avg_key_condition_emb': avg_key_condition_emb, 
            'fpaths': fpaths
        }

    ltc.s()
    path = Path('outs/250321_ipi-20u/fre_data')
    emb_npy_path_list = list(path.glob('*_embs*.npy')) # ./temp-{rank}_{idx}_embs_norm.npy

    # Sort files by rank (primary) and idx (secondary)
    sorted_emb_npy_path_list = sorted(emb_npy_path_list, key=lambda x: (
        int(x.name.split('-')[1].split('_')[0]),  # rank
        int(x.name.split('_')[1])                 # idx
    ))
    ltc.e('sort emb')

    ltc.s()
    paths = []
    f = open('250321_ipi-20u_path.txt', 'r')
    while True:
        l = f.readline()
        if not l: break
        paths.append(l.strip())
    f.close()
    ltc.e('load paths')

    ltc.s()
    fols = []
    f = open('250321_ipi-20u_fol.txt', 'r')
    while True:
        l = f.readline()
        if not l: break
        fols.append(l.strip())
    f.close()
    ltc.e('load fols')

    ltc.s()
    fol_to_datas = {}
    current_idx = 0
    current_fol = None
    make_folder('outs/250321_ipi-20u/vari_embs', remove_if_exist=True)
    f = open('outs/250321_ipi-20u/data.txt', 'w')
    for emb_path in tqdm(sorted_emb_npy_path_list, desc='loading emb..'):
        embs = np.load(emb_path)

        for emb in embs:
            current_fol = fols[current_idx]
            fpath = paths[current_idx]
            if current_fol not in fol_to_datas:
                fol_to_datas[current_fol] = []
            fol_to_datas[current_fol].append({'emb': emb, 'fpath': fpath})

            current_idx += 1

        # Check if there are any keys in fol_to_embs other than the most recent fol
        if len(fol_to_datas) > 100:
            # Get all keys except the most recent one
            keys_to_process = [k for k in fol_to_datas.keys() if k != current_fol]
            
            for key in keys_to_process:
                data = _process_fol(key, fol_to_datas[key])
                divergence_scores = data['divergence_scores']
                avg_key_condition_emb = data['avg_key_condition_emb']
                fpaths = data['fpaths']

                for fpath, divergence_score in zip(fpaths, divergence_scores):
                    f.write(f"{key.replace('/', '_')},{fpath},{divergence_score}\n")
      
                np.save(f"outs/250321_ipi-20u/vari_embs/{key.replace('/', '_')}.npy", avg_key_condition_emb)
                
            for key in keys_to_process:
                # Remove the processed key from the dictionary
                del fol_to_datas[key]

    for fol, datas in fol_to_datas.items():
        data = _process_fol(fol, datas)
        divergence_scores = data['divergence_scores']
        avg_key_condition_emb = data['avg_key_condition_emb']
        fpaths = data['fpaths']

        for fpath, divergence_score in zip(fpaths, divergence_scores):
            f.write(f"{fol.replace('/', '_')},{fpath},{divergence_score}\n")

        np.save(f"outs/250321_ipi-20u/vari_embs/{fol.replace('/', '_')}.npy", avg_key_condition_emb)
    
    f.close()
    ltc.e('save data')
   



if __name__ == '__main__':
    # temp_make_cfsc_annotation_for_fr_train(name='cfsc_50u_95d')
    # temp_make_cfsc_linkedin_annotation_for_fr_train(name='cfsc_link1M_50u_95d')

    if 0:
        last_fol_name = 'std003'
        root_path_str = f'/purestorage/AILAB/AI_1/dataset/exp_gen_data_for_fr/250402/{last_fol_name}'
        r_path = Path(root_path_str)
        jpg_path_generator = r_path.glob('**/*.jpg')
        jpg_path_list = list(jpg_path_generator)
        temp_make_annotation_for_gen_data_for_fr(jpg_path_list=jpg_path_list, name=last_fol_name, limit_imgs_per_class=99, additive_name='')
    elif 1:
        last_fol_name = 'cfsc_link30M'
        jpg_f = open('/purestorage/datasets/face_recognition/Artificial_works/variface_clip/fpath_list.txt', 'r')
        jpg_path_list = []
        line_count = 0
        while True:
            l = jpg_f.readline()
            if not l: break
            jpg_path_list.append(Path(l.strip()))
            line_count += 1
            if line_count % 10000000 == 0:
                print(f"Processed {line_count // 1000000}M lines")
        jpg_f.close()
        temp_make_annotation_for_gen_data_for_fr(jpg_path_list=jpg_path_list, name=last_fol_name, limit_imgs_per_class=99, additive_name='')

    # temp_l2_norm_process()
    # temp_extract_only_one_img_per_class()

    # temp_make_variface_annotation()


            
