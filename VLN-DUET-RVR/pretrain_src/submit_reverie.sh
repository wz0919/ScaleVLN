#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --qos=gamma_access
#SBATCH -p gamma-gpu
#SBATCH --wrap='hostname'
#SBATCH --mem=64G
#SBATCH --time=2-0:00:00

# module add cuda/11.2
# module add gcc/9.1.0
# module add nccl/1.3.3

# CUDA_VISIBLE_DEVICES=$1
# python train_hm3d_reverie.py --world_size 4 

export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1;
CUDA_VISIBLE_DEVICES=0 python -u train_hm3d_reverie.py \
    --vlnbert cmt \
    --model_config configs/model_config.json \
    --config configs/training_args.json \
    --output_dir ../datasets/REVERIE/expr_duet/pretrain/hm3d_rvr
    # --model_config config/hm3d_reverie_obj_model_config.json \
    # --config config/hm3d_reverie_obj_pretrain_text.clip.json \
    # --output_dir ../datasets/REVERIE/expr_duet/pretrain/agent7_text.clip.fix_nomlm