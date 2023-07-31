DATA_ROOT=../datasets

train_alg=dagger

features=clip.h14
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=8
seed=0

name=${train_alg}-${features}-envedit
name=${name}-seed.${seed}
name=${name}-aug.mp3d.prevalent.hm3d_gibson.envdrop.init.190k


outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}-aug.hm3d.envdrop

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size ${bs}
      --lr 1e-5
      --iters 200000
      --log_every 500
      --aug_times 9

      --env_aug

      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# # zero shot
# python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --zero_shot

# train
CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../datasets/R2R/trained_models/pretrain/duet_vit-h14_model_step_190000.pt \
      --aug ../datasets/R2R/annotations/R2R_scalevln_ft_aug_enc.json

# # test
# CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --resume_file ../datasets/R2R/trained_models/finetune/duet_vit-h14_ft_best_val_unseen \
#       --test --submit