
NODE_RANK=0
NUM_GPUS=2
outdir=../datasets/R2R/exprs_map/pretrain/cmt-clip.vit.h14-mlm.sap-init.lxmert-aug.mp3d.prevalent.hm3d_gibson.envdrop

# train
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch \
    --master_port $2 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config_clip-h14.json \
    --config config/r2r_pretrain_hm3d+mp3d+gibson_clip-h14.json \
    --output_dir $outdir 
