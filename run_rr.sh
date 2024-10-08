#!/bin/bash
ADVBENCH="AdvBench"
HARMBENCH="HarmBench"
JAILBREAKBENCH="JailbreakBench"
MALICIOUSINSTRUCT="MaliciousInstruct"
LLAMA="Llama2"
FALCON="Falcon"
VICUNA="Vicuna"
MPT="MPT"
MISTRAL="Mistral"
METHOD="RR"
NUM_BEHAVIORS=2

# python run_rr.py \
#     --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${ADVBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$ADVBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_rr.py \
#     --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${ADVBENCH}_${VICUNA}.jsonl" \
#     --model "$VICUNA" \
#     --dataset_name "$ADVBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

python run_rr.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_rr.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_rr.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

# python run_rr.py \
#     --input_file "./data/${HarmBench}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${HarmBench}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$HarmBench" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_rr.py \
#     --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$JAILBREAKBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_rr.py \
#     --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$MALICIOUSINSTRUCT" \
#     --num_behaviors "$NUM_BEHAVIORS"
