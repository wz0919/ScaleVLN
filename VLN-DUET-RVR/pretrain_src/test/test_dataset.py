import os
import sys

from data.dataset import ReverieTextPathData

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
    print(len(train_nav_db))

    print('\n\npos')
    print(train_nav_db.get_input(0, 'pos', return_act_label=True, return_img_probs=True, return_obj_label=True))

    print('\n\nneg_in_gt_path')
    print(train_nav_db.get_input(0, 'neg_in_gt_path', return_act_label=True))

    print('\n\nneg_others')
    print(train_nav_db.get_input(0, 'neg_others', return_act_label=True))
