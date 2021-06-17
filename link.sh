#!/bin/bash

for dataset in "cit-HepTh" "cit-HepPh"
do
    for feature_type in 0 1 2
    do
        save_path="./results/link/${dataset}-${feature_type}.csv"
        python link/link_pred.py -d $dataset -f $feature_type -sp $save_path
        echo $save_path
    done
done