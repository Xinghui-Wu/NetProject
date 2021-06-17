#!/bin/bash

for dataset in "cit-HepTh" "cit-HepPh"
do
    for feature_type in 0 1 2
    do
        for label_type in 1 2
        do
            save_path="./results/classification/${dataset}-${feature_type}-${label_type}.csv"
            python classification/classification.py -d $dataset -f $feature_type -l $label_type -sp $save_path
            echo $save_path
        done
    done
done