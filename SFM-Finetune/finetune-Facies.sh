CUDA_VISIBLE_DEVICES='5' OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path '../Data/Facies/' \
    --accum_iter 2 \
    --batch_size 1 \
    --model vit_base_patch16 \
    --finetune './output_dir_more/Base-512.pth'\
    --output_dir './finetune_result/SEAM/modelbase_512/' \
    --log_dir './finetune_result/SEAM/modelbase_512/' \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1.5e-3 --weight_decay 0.05 \
    --layer_decay 0.05 --drop_path 0.1 --reprob 0.25 \
    --dist_eval
    
