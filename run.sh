#!/bin/bash

CONDAENV=$(conda info | grep "active environment" | awk '{print $NF}')

if [[ "$CONDAENV" != "mt" ]]; then
    echo "Error: Active conda environment is not 'mt'. Run setup.sh first"
    exit 1
fi

python3 translate.py \
	--input_file "test.csv" \
	--text_col "text" --lang_col "language" \
	--target_language "en" --translator "easynmt" --model_name "m2m_100_1.2B" --batch_size 16 \
	--output_file "test.csv" --overwrite_output_file \
	--test --test_n 32 \
	--verbose
