#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

DROPOUTS=(0 0.2 0.4 0.6 0.8)

for drop in "${DROPOUTS[@]}"
do
    echo "Starting training with dropout $drop..."
    python tools/pytorch-examples/word_language_model/main.py \
        --data data/europarl \
        --epochs 10 \
        --dropout $drop \
        --emsize 200 --nhid 200 \
        --save models/model_drop_$drop.pt \
        --save-ppl results_drop_$drop.csv
done