import os
import sys

from data.dataset import ReverieTextPathData
from data.tasks import MlmDataset, mlm_collate

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __name__ == '__main__':
    # test
    data_dir = '/sequoia/data3/shichen/datasets'
    train_traj_files = [os.path.join(data_dir, "REVERIE/annotations/pretrain/REVERIE_train_enc.jsonl")]
    connectivity_dir = os.path.join(data_dir, "R2R/connectivity")
    img_ft_file = os.path.join(data_dir, "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5")
    obj_ft_file = os.path.join(data_dir, "REVERIE/features/obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5")
    scanvp_cands_file = os.path.join(data_dir, "R2R/annotations/scanvp_candview_relangles.json")
            
    train_nav_db = ReverieTextPathData(
        train_traj_files, img_ft_file, obj_ft_file,
        scanvp_cands_file, connectivity_dir,
        image_prob_size=1000, image_feat_size=768, angle_feat_size=4,
        obj_feat_size=768, obj_prob_size=1000,
        max_txt_len=100, max_objects=20, in_memory=True
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = MlmDataset(train_nav_db, tokenizer)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=mlm_collate)

    for batch in loader:
        for key, value in batch.items():
            print(key)
            if isinstance(value, torch.Tensor):
                print(value.size())
        break