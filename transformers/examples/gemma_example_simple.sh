#!/bin/bash
# Example: Run Gemma 3 4B with a LoRA adapter (text-only, same as test.py)

python models/gemma-3-4b-it/run.py \
    --model_path stored/gemma-lora/ \
    --device cpu \
    --user_prompt "Quote of the day: " \
    --max_tokens 50
