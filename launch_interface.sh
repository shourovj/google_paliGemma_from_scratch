#!/bin/bash

MODEL_PATH="/home/shourovj/Shourov Joarder Other materials/VLM papers/paligemmahf"
PROMPT="this is a pircture of  "
IMAGE_FILE_PATH="/home/shourovj/Shourov Joarder Other materials/VLM papers/Coding_PaliGemma/paligemma/Afridi_031.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \