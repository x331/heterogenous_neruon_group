#!/usr/bin/env bash

for seed in {0..4}
do
    sbatch end_to_end.sh $seed
done