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

# Methods=("RR" "PGD" "SoftPrompt") : Choose the appropriate attack method
METHOD="RR"

MODELS=("$LLAMA" "$FALCON" "$VICUNA" "$MPT" "$MISTRAL")
DATASETS= ("$ADVBENCH" "$HARMBENCH" "$JAILBREAKBENCH" "$MALICIOUSINSTRUCT")
NUM_BEHAVIORS=50


for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        python run_attack.py \
            --input_file "./data/${dataset}/harmful_behaviors.csv" \
            --output_file "./results/${METHOD}_${dataset}_${model}.jsonl" \
            --model "$model" \
            --attack_name "$METHOD" \
            --num_behaviors "$NUM_BEHAVIORS"
    done
done

# python run_attack.py \
#     --input_file "./data/${ADVBENCH}/harmful_behaviors.csv" \
#     --output_file "./results/${METHOD}_${ADVBENCH}_${LLAMA}.jsonl" \
#     --model "$LLAMA" \
#     --attack_name "$METHOD" \
#     --num_behaviors "$NUM_BEHAVIORS"