## Seismic Foundation Model - Pre-train

<p align="center">
  <img src="../assert/Network.png" width="480">
</p>

This is a PyTorch/GPU implementation of the paper [Seismic Foundation Model](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```
* This repo is a modification on the [MAE](https://github.com/facebookresearch/mae). Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

## Pre-training MAE

To pre-train ViT-Base/Large with **multi-node distributed training**, run the ```./train.sh``` :
```
  python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --batch_size 580\
    --accum_iter 4 \
    --model mae_vit_base_patch16D4d256 \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${DATA_DIR}
```
- Here the effective batch size is 580 (`batch_size` per gpu) * 4 (`nodes`) * 4 (gpus per node) = 9280. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 4 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
