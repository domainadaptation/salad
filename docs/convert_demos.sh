#!/bin/bash

OUTP=$2
INP=$1

for notebook in `ls $INP/*.ipynb`; do
    echo "converting" $notebook
    jupyter nbconvert --to rst --output-dir=$OUTP $notebook
done