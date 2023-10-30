#!/usr/bin/env bash

for seed in {0..4}
do
    sbatch gpu.sh $seed
done