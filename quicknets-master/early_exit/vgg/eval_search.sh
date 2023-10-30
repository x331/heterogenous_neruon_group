#!/usr/bin/env bash

for seed in {0..4}
do
    for config in 'A' 'B' 'D' 'E'
    do
        for threshold in 0.8 0.85 0.9 0.95
        do
            for lr in 0.1 0.01 0.001 0.0001
            do
                sbatch eval.sh $seed $config $threshold $lr
            done
        done
    done
done