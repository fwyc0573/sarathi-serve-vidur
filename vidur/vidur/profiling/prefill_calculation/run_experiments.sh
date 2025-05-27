#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="/research/d1/gds/ytyang/yichengfeng/vidur/sarathi-serve-vidur/vidur:$PYTHONPATH"

OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

MODELS=(
    "meta-llama/Llama-2-7b-hf"
)

# export PYTHONPATH="/research/d1/gds/ytyang/yichengfeng/vidur/sarathi-serve/sarathi:/research/d1/gds/ytyang/yichengfeng/vidur:$PYTHONPATH"


for MODEL in "${MODELS[@]}"; do
    echo "Running experiment for model: $MODEL"
    python /research/d1/gds/ytyang/yichengfeng/vidur/sarathi-serve-vidur/vidur/vidur/profiling/prefill_calculation/prefill_estimation_test.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --use_improved_method \
    --num_test_cases 1000 \
    --seed 42 \
    --measurement_runs 10 \
    --warmup_runs 5 \
    --num_tensor_parallel 1
done

echo "All experiments completed."