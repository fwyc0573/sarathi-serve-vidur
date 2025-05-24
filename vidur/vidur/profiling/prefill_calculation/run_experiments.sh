#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7


OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

MODELS=(
    "meta-llama/Llama-2-7b-hf"
)

# export PYTHONPATH="/research/d1/gds/ytyang/yichengfeng/vidur/sarathi-serve/sarathi:/research/d1/gds/ytyang/yichengfeng/vidur:$PYTHONPATH"


for MODEL in "${MODELS[@]}"; do
    echo "Running experiment for model: $MODEL"
    python /research/d1/gds/ytyang/yichengfeng/vidur/sarathi-serve-vidur/vidur/vidur/profiling/prefill_calculation/prefill_estimation_test.py --model $MODEL --output_dir $OUTPUT_DIR
done

echo "All experiments completed."