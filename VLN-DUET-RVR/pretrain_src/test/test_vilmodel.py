import os
import sys
import json
from easydict import EasyDict

from data.dataset import ReverieTextPathData
from data.tasks import MlmDataset, mlm_collate
from model.vilmodel import GlocalTextPathCMT

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig

if __name__ == '__main__':
    # test
    data_dir = '/sequoia/data3/shichen/datasets'
    train_traj_files = [os.path.join(data_dir, "REVERIE/annotations/pretrain/REVERIE_train_enc.jsonl")]
    connectivity_dir = os.path.join(data_dir, "R2R/connectivity")
    img_ft_file = os.path.join(data_dir, "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5")
    obj_ft_file = os.path.join(data_dir, "REVERIE/features/obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5")
    scanvp_cands_file = os.path.join(data_dir, "R2R/annotations/scanvp_candview_relangles.json")

    model_config = PretrainedConfig.from_json_file('config/reverie_obj_model_config.json')
    model = GlocalTextPathCMT(model_config).cuda()
            
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
                batch[key] = value.cuda()
                print('\t', value.size())
        txt_embeds = model.forward_mlm(
            batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
            batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
            batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
            batch['traj_vpids'], batch['traj_cand_vpids'], 
            batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
            batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
        )
        print(txt_embeds)
        s = txt_embeds.sum()
        s.backward()
        print(s)
