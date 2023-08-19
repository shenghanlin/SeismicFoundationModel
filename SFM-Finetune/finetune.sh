OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path '/home/hlsheng/mae_data/finetune/SEAM_Faces/' \
    --accum_iter 2 \
    --batch_size 1 \
    --model vit_large_patch16 \
    --finetune './output_dir_more/Large-1600.pth' \
    --output_dir './finetune_faces_sert3_large800/' \
    --log_dir './finetune_faces_sert3_large800/' \
    --epochs 500 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 \
    --dist_eval
    
