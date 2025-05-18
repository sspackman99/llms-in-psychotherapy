#!/bin/bash
DATA_PATH="Data/share.csv"
NUM_EXAMPLES=2
SAMPLES_PATH="Data/samples.csv"
TEMPERATURE=0.7
TOP_P=0.9
MAX_SEQ_LEN=4960
MAX_GEN_LEN=4960
MAX_BATCH_SIZE=1

OUTPUT_FILE="output.txt"
ERROR_FILE="errors.txt"

python python_scripts/anonymize.py \
    $DATA_PATH \
    $NUM_EXAMPLES \
    $SAMPLES_PATH \
    $TEMPERATURE \
    $TOP_P \
    $MAX_SEQ_LEN \
    $MAX_GEN_LEN \
    $MAX_BATCH_SIZE \
    > $OUTPUT_FILE 2> $ERROR_FILE
