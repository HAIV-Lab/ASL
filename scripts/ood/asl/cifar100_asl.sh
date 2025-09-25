export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/asl_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_asl.yml \
    --trainer.use_amp False \
    --output_dir results \
    --network.backbone.name resnet18_32x32 \
    --dataset.train.batch_size 128 \
    --num_workers 4 \
    --optimizer.num_epochs 100 \
    --seed 0

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_asl_net_asl_e100_lr0.1_default \
   --postprocessor asl \
   --save-csv
