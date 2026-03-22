
# MT Exercise 2: Pytorch RNN Language Models (Europarl)

This repository shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). It has been adapted for **Exercise 02** to experiment with Dropout rates and the Europarl dataset.

## Requirements

- This only works on a Unix-like system with `bash`.
- Python 3 must be installed (the command `python3` must be available).
- `virtualenv` must be installed: `pip install virtualenv`.

## Steps

### 1. Setup Environment
Clone this repository and create a new virtualenv:
```bash
git clone [https://github.com/Joyoem/mt-exercise-02](https://github.com/Joyoem/mt-exercise-02)
cd mt-exercise-02
./scripts/make_virtualenv.sh
```
**Important**: Activate the environment by executing the `source` command output by the script.

### 2. Install Packages
Download and install required software:
```bash
./scripts/install_packages.sh
```

### 3. Data Preparation
The following script has been modified to download and preprocess the **Europarl** dataset (instead of the default data):
```bash
./scripts/download_data.sh
```
#### Detailed File Modifications
- Source Change: Migrated from PTB to Europarl v7 corpus.

- Data Cleaning: Implemented grep filters to strip XML metadata tags from raw proceedings.

- Preprocessing Pipeline: Integrated preprocess.py with specific flags:

  - tokenize: To handle formal English punctuation and contractions.

  - vocab-size 10000: To constrain the model complexity (leading to <unk> tokens for rare words).

- Custom Splitting: Hardcoded fixed line ranges for Train (70k lines), Valid (5k lines), and Test (5k lines) to ensure reproducible perplexity metrics.

### 4. Model training for part1

```bash
./scripts/train.sh
```
#### Detailed File Modifications

- Path Adaptation: Updated to utilize the preprocessed Europarl dataset.

- Efficiency Tuning: Reduced training intensity to 20 epochs because each epoch costs about >10min, 40*10 is too lang.

- Logging: Implemented a tee pipeline to preserve all terminal outputs into train_log.txt, which served as the primary data source for manual Test PPL extraction.

### 5. sample generation for part1
```bash
./scripts/generate.sh
```
#### Detailed File Modifications

- Path Alignment: Updated `--data` and `--checkpoint` paths to ensure the generator uses the Europarl vocabulary and the correct model weights.
- Output Archiving: Added the `--outf` flag to automatically save generated text to the `samples/` directory for qualitative analysis.

  

### 6. Code Modifications & New Scripts for part2

- **`tools/pytorch-examples/word_language_model/main.py` (Modified)**:
    * **New Flag**: Added `--dropout` to the argument parser to allow external control.
    * **Unique Checkpoints**: Updated the `--save` logic to include the dropout value in the filename (e.g., `model_drop_0.4.pt`), preventing experimental data from being overwritten.
     
- **`scripts/train_dropout.sh` (New)**:
    * A high-level automation script that executes a loop to train 5 separate models with dropout rates: `0, 0.2, 0.4, 0.6, 0.8`.
      
- **`scripts/plot_results.py` (New)**:
    * **Data Aggregation**: Parses the training logs to generate PPL curves.
    * **Manual Data Entry**: Since the baseline CSV logging does not capture Test PPL, I manually extracted the final Test PPL values from the terminal output (saved via `tee`) and hardcoded them into this script to produce the final comparison tables.
      
- **`scripts/generate_task2.sh` (New)**:
    * Automates text sampling specifically from the **best-performing** (Drop 0) and **worst-performing** (Drop 0.8) models for direct qualitative comparison.

#### Execution Workflow
run the following commands in order:

1.   
    ```bash
    bash scripts/train_dropout.sh
    ```
2.  Visualization: 
    ```bash
    python scripts/plot_results.py
    ```
3.  Comparison: 
    ```bash
    bash scripts/generate_task2.sh
    ```

#### Note
While the scripts generate logs, CSVs, and PNGs in the root directory by default, these have been manually organized into the `results/` folder for clarity:
