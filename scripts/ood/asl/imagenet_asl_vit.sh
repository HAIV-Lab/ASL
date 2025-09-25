export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/asl_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_asl.yml \
    --trainer.use_amp True \
    --network.backbone.name vit-b-16 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./z_results/vit-b-16/vit_b_16-c867db91.pth \
    --optimizer.lr 0.00001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 1 --num_workers 8 \
    --merge_option merge \
    --seed 0

python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet_asl_net_asl_e90_lr0.001_default \
   --postprocessor asl \
   --save-csv
