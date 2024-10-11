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
METHOD="AmpleGCG"
NUM_BEHAVIORS=50

# python run_ample_gcg.py \
#     --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${ADVBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$ADVBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_ample_gcg.py \
#     --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${HARMBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$HARMBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_ample_gcg.py \
#     --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$JAILBREAKBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_ample_gcg.py \
#     --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --dataset_name "$MALICIOUSINSTRUCT" \
#     --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${VICUNA}.jsonl" \
    --model "$VICUNA" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${HARMBENCH}_${VICUNA}.jsonl" \
    --model "$VICUNA" \
    --dataset_name "$HARMBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${VICUNA}.jsonl" \
    --model "$VICUNA" \
    --dataset_name "$JAILBREAKBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${VICUNA}.jsonl" \
    --model "$VICUNA" \
    --dataset_name "$MALICIOUSINSTRUCT" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${HARMBENCH}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$HARMBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$JAILBREAKBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$MALICIOUSINSTRUCT" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${HARMBENCH}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$HARMBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$JAILBREAKBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$MALICIOUSINSTRUCT" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${HARMBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$HARMBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$JAILBREAKBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_ample_gcg.py \
    --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$MALICIOUSINSTRUCT" \
    --num_behaviors "$NUM_BEHAVIORS"