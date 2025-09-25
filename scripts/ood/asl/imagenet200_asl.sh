export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/asl_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_asl.yml \
    --trainer.use_amp True \
    --network.backbone.name resnet18_224x224 \
    --dataset.train.batch_size 128 \
    --num_gpus 1 --num_workers 8 \
    --optimizer.num_epochs 90 \
    --seed 0

python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_asl_net_asl_e90_lr0.1_default \
   --postprocessor asl \
   --save-csv
