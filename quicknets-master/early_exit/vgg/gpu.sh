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
lr=${3:-0.01}
echo $seed $config $lr

python train.py --seed $seed --save_params --config $config --lr $lr --batch_norm
exit