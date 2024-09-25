#!/bin/bash
ADVBENCH="AdvBench"
HARMBENCH="HarmBench"
JAILBREAKBENCH="JailbreakBench"
MALICIOUSINSTRUCT="MaliciousInstruct"
MODEL="Llama2"
METHOD="GCG"
NUM_BEHAVIORS=50
# INPUT_FILE="/home/samuel/Research/reg_relax/data/${DATASETS[0]}/harmful_behaviors.csv"
# OUTPUT_FILE="/home/samuel/Research/reg_relax/results/${METHOD}_${DATASET_NAME}_${MODEL}.jsonl"

python run_gcg.py \
    --input_file "/home/samuel/Research/reg_relax/data/${ADVBENCH}/harmful_behaviors.csv" \
    --output_file "/home/samuel/Research/reg_relax/results/${METHOD}_${ADVBENCH}_Falcon.jsonl" \
    --dataset_name "$ADVBENCH" \
    --num_behaviors "$NUM_BEHAVIORS"

# python run_gcg.py \
#     --input_file "/home/samuel/Research/reg_relax/data/${HARMBENCH}/harmful_behaviors.csv" \
#     --output_file "/home/samuel/Research/reg_relax/results/${METHOD}_${HARMBENCH}_Falcon.jsonl" \
#     --dataset_name "$HARMBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"

# python run_gcg.py \
#     --input_file "/home/samuel/Research/reg_relax/data/${ADVBENCH}/harmful_behaviors.csv" \
#     --output_file "/home/samuel/Research/reg_relax/results/${METHOD}_${ADVBENCH}_${MODEL}.jsonl" \
#     --dataset_name "$ADVBENCH" \
#     --num_behaviors "$NUM_BEHAVIORS"