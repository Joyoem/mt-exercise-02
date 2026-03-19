#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

# list
mkdir -p $data/europarl/raw

# download
wget -N https://www.statmt.org/europarl/v7/europarl.tgz
tar -xvzf europarl.tgz -C $data/europarl/raw/ txt/

# combine and clean
cat $data/europarl/raw/txt/en/ep-11-*.txt | grep -v '^<' > $data/europarl/raw/europarl_cleaned.txt

# preprocessing
cat $data/europarl/raw/europarl_cleaned.txt | python3 $base/scripts/preprocess.py \
    --vocab-size 10000 --tokenize --lang "en" --sent-tokenize > \
    $data/europarl/raw/europarl.preprocessed.txt


# dividing dataset
# validation set
head -n 5000 $data/europarl/raw/europarl.preprocessed.txt > $data/europarl/valid.txt

# test set
head -n 10000 $data/europarl/raw/europarl.preprocessed.txt | tail -n 5000 > $data/europarl/test.txt

# training set
sed -n '10001,80000p' $data/europarl/raw/europarl.preprocessed.txt > $data/europarl/train.txt

# final delete
rm -rf $data/europarl/raw/txt/


 