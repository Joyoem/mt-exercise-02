#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

cd $tools/pytorch-examples/word_language_model

# Lowest PPL Model (Dropout 0)
echo "Generating from Dropout 0 model..."
python generate.py \
    --data $base/data/europarl \
    --words 100 \
    --checkpoint $models/model_drop_0.pt \
    --outf $samples/sample_best.txt

# Highest PPL Model (Dropout 0.8)
echo "Generating from Dropout 0.8 model..."
python generate.py \
    --data $base/data/europarl \
    --words 100 \
    --checkpoint $models/model_drop_0.8.pt \
    --outf $samples/sample_worst.txt

echo "Comparison samples generated in $samples/"