CUDA_VISIBLE_DEVICES='0' OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path '../mae_data/train/' \
    --task 'Interpolation' \
    --accum_iter 1 \
    --batch_size 50 \
    --input_size 224 \
    --model vit_base_patch16 \
    --resume './finetune_result/Interpolation/net_Transformer_16_800epoch/checkpoint-99.pth' \
    --output_dir './finetune_result/Interpolation/net_Transformer_16_800epoch/' \
    --log_dir './finetune_result/Interpolation/net_Transformer_16_800epoch/' \
    --epochs 800 \
    --warmup_epochs 30 \
    --blr 1.5e-4 --weight_decay 0.75 \
    --layer_decay 0.5 --drop_path 0.1 --reprob 0.25 \
    --dist_eval \
    
