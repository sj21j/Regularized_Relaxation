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
METHOD="GCG"
NUM_BEHAVIORS=50

python run_gcg.py \
    --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${ADVBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_gcg.py \
    --input_file "./data/${HARMBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${HARMBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$HARMBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_gcg.py \
    --input_file "./data/${JAILBREAKBENCH}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${JAILBREAKBENCH}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$JAILBREAKBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

python run_gcg.py \
    --input_file "./data/${MALICIOUSINSTRUCT}/harmful_behaviors.csv" \
    --output_file "./results/${METHOD}_${MALICIOUSINSTRUCT}_${MISTRAL}.jsonl" \
    --model "$MISTRAL" \
    --dataset_name "$MALICIOUSINSTRUCT" \
    --num_behaviors "$NUM_BEHAVIORS"
