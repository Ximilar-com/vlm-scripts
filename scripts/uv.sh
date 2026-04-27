#!/bin/bash

# THIS SCRIPT IS NOT INTENDED TO BE RUN DIRECTLY, BUT RATHER AS A TEMPLATE FOR RUNNING THE INDIVIDUAL SCRIPTS

uv run transformers/models/Qwen3-VL-2B-Instruct/run.py \
    --model_path $HOME/Downloads/model/ \
    --images $HOME/Downloads/google.webp \
    --user_prompt "Describe this image." \
    --system_prompt "You are a helpful assistant for answering questions about the image. Answer in detail and be as descriptive as possible."

uv run transformers/models/LFM2.5-VL-1.6B/run.py \
    --model_path $HOME/Downloads/model/ \
    --images $HOME/Downloads/google.webp \
    --device cpu \
    --user_prompt "Describe this image." \
    --system_prompt "You are a helpful assistant for answering questions about the image. Answer in detail and be as descriptive as possible."
