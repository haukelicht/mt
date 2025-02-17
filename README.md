# Machine translation in python

## Setup 

1. run (code in) setup.sh
2. `chmod +x translate.py`

## Usage

- `--input_file`: Input file
- `--output_file`: Output file. If not specified and --overwrite_output_file is set, input file will be overwritten.
- `--overwrite_output_file`: Overwrite output file if it exists. If output file not specified, input file will be overwritten.
- `--text_col`: Name of column containing texts to be translated
- `--lang_col`: Name of column that indicates to-be-translated texts' language codes
- `--target_language`: ISO 639-1 language code of "target" language to translate texts to
- `--overwrite_target_column`: Overwrite target column if it exists.
- `--translator`: Translator to use. One of "easynmt", "google", or "deepl".
- `--model_name`: Name of model to use for translation. Only used if translator is "easynmt". See https://github.com/UKPLab/EasyNMT#available-models for available models.
- `--api_key_file`: Path to file containing API key. Only used if translator is "google" or "deepl".
- `--batch_size`: Batch size for translation. Recommended to set to 32 or lower if translator "easynmt" is used with GPU.
- `--split_sentences`: Split sentences before translation. CAUTION: Sentence splitting in supported models is punctuation based and might be wrong, potentially impairing translation quality.
- `--verbose`: Print progress bar and other messages.
- `--test`: Run in test mode.
- `--test_n`: number of texts to sample per language in test mode

### Example

*Note:* see also [`run.sh`](./run.sh)

```bash
python3 translate.py \
	--input_file "test.csv" \ # allows CSV or tab-separated (.tev or .tab)
	--text_col "text" --lang_col "language" \
	--target_language "en" --translator "easynmt" --model_name "m2m_100_1.2B" --batch_size 16 \
	--output_file "test.csv" --overwrite_output_file \
	--test --test_n 32 \ # tets mode samples up to `test_n` examples per language from input file
	--verbose
```
