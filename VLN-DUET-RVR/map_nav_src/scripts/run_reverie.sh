#! /bin/bash

DATA_ROOT=../datasets

train_alg=dagger

features=clip-h14
ft_dim=1024
obj_features=timm_vitb16
obj_ft_dim=768

ngpus=1
seed=0

name=scalevln_rvr
outdir=${DATA_ROOT}/REVERIE/expr_duet/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 100
      --max_objects 50

      --batch_size 32
      --lr 2e-5
      --iters 200000
      --log_every 500
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

CUDA_VISIBLE_DEVICES=0 python reverie/main_nav_obj_hm3d.py $flag --tokenizer bert \
      --bert_ckpt_file ../datasets/REVERIE/trained_models/model_step_40000.pt \
      --aug ../datasets/REVERIE/annotations/ade20k_pseudo3d_depth2_epoch_94_beam0_sample10.jsonl \
      --eval_first 

# python reverie/main_nav_obj_hm3d.py $flag --tokenizer bert \
#       --zero_shot 