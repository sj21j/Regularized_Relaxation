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
METHOD="PGD"
NUM_BEHAVIORS=2


python run_pgd.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${LLAMA}.jsonl" \
    --model "$LLAMA" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_pgd.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${VICUNA}.jsonl" \
    --model "$VICUNA" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_pgd.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${FALCON}.jsonl" \
    --model "$FALCON" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_pgd.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MPT}.jsonl" \
    --model "$MPT" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_pgd.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"