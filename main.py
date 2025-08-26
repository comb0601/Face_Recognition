
import argparse
from enum import Enum
from typing import Literal

from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader

from utils import cosine_similarity, setup_distributed_training_in_slurm, CustomDistributedSampler, make_folder
from dataset import CUBOX_RGB_FAS, CropFaceSet
from job import FR_Embedding_Extractor, FaceClusterProcessor

class JOB(Enum):
    FR_EXTRACT = 'fre'
    FACE_CLUSTER = 'fc'

class Config(BaseModel):
    fre_run_mode: Literal['fre', 'variface']
    fre_dataset_kwargs: dict
    fre_emb_save_path: str
    fre_batch_size: int
    fre_use_l2_norm: bool
    fre_emb_dim: int
    fre_merge_embs: bool

    fc_run_mode: Literal['all', 'index_search', 'process_search_result', 'label_for_fr']
    fc_do_l2_norm: bool
    fc_emb_npy_path_list: list[str]
    fc_fpath_npy_path_list: list[str]
    fc_save_fol_path: str
    fc_same_person_cos_threshold: float
    fc_same_image_cos_threshold: float
    fc_topK: int
    fc_threshold_num_emb_to_split: int
    fc_make_sample_folder: bool
    fc_make_fr_label_for_cfsc_and_linkedin: bool
DATA_LIST = {
    'cfsc': (['outs/cfsc_all/final_embs_norm.npy'], ['outs/cfsc_all/final_fpaths.npy']),
    'linked-28M': (['outs/merged_lin-r-v1-v11_refined/embs_norm.npy'], ['outs/merged_lin-r-v1-v11_refined/fpaths.npy']),

    'v1': ([f'outs/linkedin_v1/fre_data/embs_norm_{i}.npy' for i in range(39)], [f'outs/linkedin_v1/fre_data/fpaths_{i}.npy' for i in range(39)]),
    'v2': ([f'outs/linkedin_v2/fre_data/embs_norm_{i}.npy' for i in range(37)], [f'outs/linkedin_v2/fre_data/fpaths_{i}.npy' for i in range(37)]),
    'v3': ([f'outs/linkedin_v3/fre_data/embs_norm_{i}.npy' for i in range(34)], [f'outs/linkedin_v3/fre_data/fpaths_{i}.npy' for i in range(34)]),
    'v4': ([f'outs/linkedin_v4/fre_data/embs_norm_{i}.npy' for i in range(34)], [f'outs/linkedin_v4/fre_data/fpaths_{i}.npy' for i in range(34)]),
    'v5': ([f'outs/linkedin_v5/fre_data/embs_norm_{i}.npy' for i in range(34)], [f'outs/linkedin_v5/fre_data/fpaths_{i}.npy' for i in range(34)]),
    'v6': ([f'outs/linkedin_v6/fre_data/embs_norm_{i}.npy' for i in range(33)], [f'outs/linkedin_v6/fre_data/fpaths_{i}.npy' for i in range(33)]),
    'v7': ([f'outs/linkedin_v7/fre_data/embs_norm_{i}.npy' for i in range(32)], [f'outs/linkedin_v7/fre_data/fpaths_{i}.npy' for i in range(32)]),
    'v8': ([f'outs/linkedin_v8/fre_data/embs_norm_{i}.npy' for i in range(31)], [f'outs/linkedin_v8/fre_data/fpaths_{i}.npy' for i in range(31)]),
    'v9': ([f'outs/linkedin_v9/fre_data/embs_norm_{i}.npy' for i in range(31)], [f'outs/linkedin_v9/fre_data/fpaths_{i}.npy' for i in range(31)]),
    'v10': ([f'outs/linkedin_v10/fre_data/embs_norm_{i}.npy' for i in range(30)], [f'outs/linkedin_v10/fre_data/fpaths_{i}.npy' for i in range(30)]),
    # 'v11': ([f'outs/linkedin_v11/fre_data/embs_norm_{i}.npy' for i in range(26)], [f'outs/linkedin_v11/fre_data/fpaths_{i}.npy' for i in range(26)]),
    'v11': (['outs/linkedin_v11/fre_data/final_embs_norm.npy' for i in range(1)], ['outs/linkedin_v11/fre_data/final_fpaths.npy' for i in range(1)]),
    
    'v1-r': (['outs/linkedin_v1_refined/embs_norm.npy'], ['outs/linkedin_v1_refined/fpaths.npy']),
    'v2-r': (['outs/linkedin_v2_refined/embs_norm.npy'], ['outs/linkedin_v2_refined/fpaths.npy']),
    'v3-r': (['outs/linkedin_v3_refined/embs_norm.npy'], ['outs/linkedin_v3_refined/fpaths.npy']),
    'v4-r': (['outs/linkedin_v4_refined/embs_norm.npy'], ['outs/linkedin_v4_refined/fpaths.npy']),
    'v5-r': (['outs/linkedin_v5_refined/embs_norm.npy'], ['outs/linkedin_v5_refined/fpaths.npy']),
    'v6-r': (['outs/linkedin_v6_refined/embs_norm.npy'], ['outs/linkedin_v6_refined/fpaths.npy']),
    'v7-r': (['outs/linkedin_v7_refined/embs_norm.npy'], ['outs/linkedin_v7_refined/fpaths.npy']),
    'v8-r': (['outs/linkedin_v8_refined/embs_norm.npy'], ['outs/linkedin_v8_refined/fpaths.npy']),
    'v9-r': (['outs/linkedin_v9_refined/embs_norm.npy'], ['outs/linkedin_v9_refined/fpaths.npy']),
    'v10-r': (['outs/linkedin_v10_refined/embs_norm.npy'], ['outs/linkedin_v10_refined/fpaths.npy']),
    'v11-r': (['outs/linkedin_v11_refined/embs_norm.npy'], ['outs/linkedin_v11_refined/fpaths.npy']),
}

PRESET_DICT = {
    'CFSC': {
        'embs': DATA_LIST['cfsc'][0], 
        'fpaths': DATA_LIST['cfsc'][1],
        'save_fol': 'cfsc_all_revision_v2'
    },
    'linked-v1': {
        'embs': DATA_LIST['v1'][0], 
        'fpaths': DATA_LIST['v1'][1],
        'search_cos': [f'outs/linkedin_v1/search/cosine_{i}.npy' for i in range(8)],
        'search_idx': [f'outs/linkedin_v1/search/index_{i}.npy' for i in range(8)],
        'save_fol': 'linkedin_v1'
    },
    'linked-v2': {
        'embs': DATA_LIST['v2'][0], 
        'fpaths': DATA_LIST['v2'][1],
        'save_fol': 'linkedin_v2'
    },
    'linked-v3': {
        'embs': DATA_LIST['v3'][0], 
        'fpaths': DATA_LIST['v3'][1],
        'save_fol': 'linkedin_v3'
    },
    'linked-v4': {
        'embs': DATA_LIST['v4'][0], 
        'fpaths': DATA_LIST['v4'][1],
        'save_fol': 'linkedin_v4'
    },
    'linked-v5': {
        'embs': DATA_LIST['v5'][0], 
        'fpaths': DATA_LIST['v5'][1],
        'save_fol': 'linkedin_v5'
    },
    'linked-v6': {
        'embs': DATA_LIST['v6'][0], 
        'fpaths': DATA_LIST['v6'][1],
        'save_fol': 'linkedin_v6'
    },
    'linked-v7': {
        'embs': DATA_LIST['v7'][0], 
        'fpaths': DATA_LIST['v7'][1],
        'save_fol': 'linkedin_v7'
    },
    'linked-v8': {
        'embs': DATA_LIST['v8'][0], 
        'fpaths': DATA_LIST['v8'][1],
        'save_fol': 'linkedin_v8'
    },
    'linked-v9': {
        'embs': DATA_LIST['v9'][0], 
        'fpaths': DATA_LIST['v9'][1],
        'save_fol': 'linkedin_v9'
    },
    'linked-v10': {
        'embs': DATA_LIST['v10'][0], 
        'fpaths': DATA_LIST['v10'][1],
        'save_fol': 'linkedin_v10'
    },
    'linked-v11': {
        'embs': DATA_LIST['v11'][0], 
        'fpaths': DATA_LIST['v11'][1],
        'save_fol': 'linkedin_v11'
    },
    'linked-r-v1-v11': {
        'embs': [item for v_name in [f'v{i}-r' for i in range(1,12)] for item in DATA_LIST[v_name][0]], 
        'fpaths': [item for v_name in [f'v{i}-r' for i in range(1,12)] for item in DATA_LIST[v_name][1]],
        'save_fol': 'merged_lin-r-v1-v11'
    },
    'CFSC_linked-v1': {
        'embs': DATA_LIST['cfsc'][0] + DATA_LIST['v1'][0], 
        'fpaths': DATA_LIST['cfsc'][1] + DATA_LIST['v1'][1],
        'search_cos': [f'outs/merged_cfsc_linv1_revision3/search/cosine_{i}.npy' for i in range(6)],
        'search_idx': [f'outs/merged_cfsc_linv1_revision3/search/index_{i}.npy' for i in range(6)],
        'save_fol': 'merged_cfsc_linv1_revision3_2'
    },
    'CFSC_linked-28M': {
        'embs': DATA_LIST['cfsc'][0] + DATA_LIST['linked-28M'][0], 
        'fpaths': DATA_LIST['cfsc'][1] + DATA_LIST['linked-28M'][1],
        'save_fol': 'merged_cfsc_lin-28M_v4'
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--preset', type=str, default='linked-v11')
    parser.add_argument('--preset', type=str, default='CFSC_linked-28M')
    parser.add_argument('--job', type=str, default=JOB.FR_EXTRACT.value)
    #parser.add_argument('--job', type=str, default=JOB.FACE_CLUSTER.value)
    
    parser.add_argument('--fre_run_mode', type=str, default='fre')
    parser.add_argument('--fc_run_mode', type=str, default='all')
    #parser.add_argument('--fc_run_mode', type=str, default='label_for_fr')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1)
    args = parser.parse_args()

    PRESET = PRESET_DICT[args.preset]

    cfg = Config(
        fre_run_mode=args.fre_run_mode,
        fre_dataset_kwargs={
            # 'root_path': '/purestorage/AILAB/AI_1/dataset/linkedin_faces/imgs',
            # 'path_file': '/purestorage/AILAB/AI_1/dataset/linkedin_faces/v11_jpg_paths.txt',
            # 'path_file' : '/purestorage/datasets/face_recognition/CFSC_data/all_jpg_paths.txt',
            'path_file': './code/5M_127M_data_list.txt'
        },
        fre_emb_save_path='./extrac_FULL',
        # fre_emb_save_path='./outs/250321_ipi-20u/fre_data',
        fre_batch_size=16384,
        fre_use_l2_norm=True,
        fre_emb_dim=512,
        fre_merge_embs=True,

        fc_run_mode=args.fc_run_mode,
        fc_do_l2_norm=False,
        fc_emb_npy_path_list=PRESET['embs'],
        fc_fpath_npy_path_list=PRESET['fpaths'],
        fc_save_fol_path=f"outs/{PRESET['save_fol']}",
        fc_same_person_cos_threshold=0.5, # bigger than this -> will be in same person group
        fc_same_image_cos_threshold=0.90, # float|None ## bigger than this -> one side of pair will be excluded
        fc_topK=100,
        fc_threshold_num_emb_to_split=1000*10000, # GPU: 10M = 52GB, 1M = 8GB
        fc_make_sample_folder=False,
        fc_make_fr_label_for_cfsc_and_linkedin=True,
    )

    

    if args.job == JOB.FR_EXTRACT.value:
        fr_extractor = FR_Embedding_Extractor()

        if cfg.fre_run_mode == 'fre':
            #local_rank, global_rank, node_rank, world_size = setup_distributed_training_in_slurm()
            dist_info = {
                'local_rank': 0,
                'global_rank': 0,
                'node_rank': 0,
                'world_size': 8
                # 'local_rank': local_rank,
                # 'global_rank': global_rank,
                # 'node_rank': node_rank,
                # 'world_size': world_size,r
            }
            #source_dataset = CUBOX_RGB_FAS()
            
            source_dataset = CropFaceSet(**cfg.fre_dataset_kwargs)
            source_dataloader = DataLoader(
                source_dataset, 
                batch_size=cfg.fre_batch_size, 
                # sampler=CustomDistributedSampler(source_dataset, shuffle=False, is_test=True),
                shuffle=False, 
                drop_last=False, 
                num_workers=16)
            
            fr_extractor.run(
                source_dataloader=source_dataloader, 
                save_path=cfg.fre_emb_save_path,
                use_l2_norm=cfg.fre_use_l2_norm, 
                emb_dim=cfg.fre_emb_dim,
                merge_embs=cfg.fre_merge_embs
            )
        elif cfg.fre_run_mode == 'variface':
            fr_extractor.run_variface()
    
    if args.job == JOB.FACE_CLUSTER.value:
        face_cluster_processor = FaceClusterProcessor(
            # emb_npy_path_list=cfg.fc_emb_npy_path_list if cfg.fc_run_mode in ['all', 'index_search'] else None, 
            #emb_npy_path_list = ['outs/linkedin_v11/fre_data/final_embs_norm.npy'],
            emb_npy_path_list = ['/purestorage/AILAB/AI_1/hkl/project/process_about_fr_outs/cfsc_all/final_embs_norm.npy'],
            fpath_npy_path_list = ['/purestorage/AILAB/AI_1/hkl/project/process_about_fr_outs/cfsc_all/final_fpaths.npy'],
            # fpath_npy_path_list=None if cfg.fc_run_mode == 'index_search' else cfg.fc_fpath_npy_path_list, 
            use_gpu=True,
            do_l2_norm=cfg.fc_do_l2_norm,
            same_person_cos_threshold=cfg.fc_same_person_cos_threshold,
            same_image_cos_threshold=cfg.fc_same_image_cos_threshold,
        )

        if cfg.fc_run_mode == 'all':
            face_cluster_processor.run(
                topK=cfg.fc_topK, 
                out_root_path=cfg.fc_save_fol_path, 
                #make_sample_folder=cfg.fc_make_sample_folder,
                make_sample_folder=False,
                threshold_num_emb_to_split=cfg.fc_threshold_num_emb_to_split,
                make_fr_label_for_cfsc_and_linkedin=cfg.fc_make_fr_label_for_cfsc_and_linkedin,
            )

        if cfg.fc_run_mode == 'index_search':
            if args.node_rank == 0:
                make_folder(f'{cfg.fc_save_fol_path}/search', remove_if_exist=False)

            face_cluster_processor.run_index_search(
                topK=cfg.fc_topK, 
                out_root_path=cfg.fc_save_fol_path, 
                threshold_num_emb_to_split=cfg.fc_threshold_num_emb_to_split,
                node_rank=args.node_rank,
                nnodes=args.nnodes,
            )
        
        if cfg.fc_run_mode == 'process_search_result':
            make_folder(f'{cfg.fc_save_fol_path}', remove_if_exist=False)
            face_cluster_processor.run_get_same_person_group_
            face_cluster_processor(
                out_root_path=cfg.fc_save_fol_path, 
                search_cos_npy_path_list=PRESET['search_cos'],
                search_idx_npy_path_list=PRESET['search_idx'],
                make_sample_folder=cfg.fc_make_sample_folder,
                make_fr_label_for_cfsc_and_linkedin=cfg.fc_make_fr_label_for_cfsc_and_linkedin,
            )

        if cfg.fc_run_mode == 'label_for_fr':
            face_cluster_processor.run_make_label_for_fr_train(
                out_root_path=cfg.fc_save_fol_path,
                indice_group_txt_path=f'{cfg.fc_save_fol_path}/indice_group.txt'
            )
