#!/bin/bash
DATA_PATH="outputs/anon_2share_random_appointID.csv"
NUM_EXAMPLES=8
SAMPLES_PATH="Data/cleaned_labled_samples.csv"
TEMPERATURE=0.7
TOP_P=0.9
MAX_SEQ_LEN=4960
MAX_GEN_LEN=4960
MAX_BATCH_SIZE=1

OUTPUT_FILE="extract_output.txt"
ERROR_FILE="extract_errors.txt"

python python_scripts/extract_data.py \
    $DATA_PATH \
    $NUM_EXAMPLES \
    $SAMPLES_PATH \
    $TEMPERATURE \
    $TOP_P \
    $MAX_SEQ_LEN \
    $MAX_GEN_LEN \
    $MAX_BATCH_SIZE \
    > $OUTPUT_FILE 2> $ERROR_FILE
