#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=00-04:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
config=${2:-A}
threshold=${3:-0.8}
lr=${4:-0.1}

echo $seed $config

python eval.py --seed $seed --config $config --dataset CIFAR10 --balance_decision_layer --compressed_classifier --threshold $threshold --method decision --train_method backbone_first --lr $lr
exit