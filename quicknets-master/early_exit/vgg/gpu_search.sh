#!/usr/bin/env bash

for seed in {0..0}
do
    for config in 'D'
    do
        for lr in 0.003
        do
            sbatch gpu.sh $seed $config $lr
        done
    done
done