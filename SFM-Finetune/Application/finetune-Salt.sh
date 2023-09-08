CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path ../Data/Geobody' \
    --task 'Salt' \
    --accum_iter 1 \
    --batch_size 64 \
    --input_size 224 \
    --model vit_large_patch16 \
    --finetune './output_dir_more/Large-1600.pth' \
    --output_dir './finetune_result/Salt/sert_Transformer_test/' \
    --log_dir './finetune_result/Salt/sert_Transformer_test/' \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1.5e-3 --weight_decay 0.05 \
    --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 \
    --dist_eval
    
