# /research/d1/gds/ytyang/yichengfeng/vidur/limitation-check/prefill-calculation/prefill_estimation_test.py

import argparse
import datetime
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import AttentionBackend

# Import from vidur codebase
from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.utils import get_max_num_blocks


def parse_args():
    parser = argparse.ArgumentParser(description="Prefill Estimation Accuracy Test")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model to test with",
    )
    parser.add_argument(
        "--num_tensor_parallel",
        type=int,
        default=1,
        help="Number of tensor parallel workers",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum context length model can serve",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",
    )
    parser.add_argument(
        "--attention_backend",
        default=AttentionBackend.FLASHINFER,
        choices=[e.value for e in AttentionBackend],
        help="The attention backend to profile",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    return parser.parse_args()


def generate_test_cases() -> List[List[int]]:
    """Generate diverse test cases of different prefill length combinations."""
    # Define some interesting test cases
    test_cases = [
        # Equal length batches
        [128, 128],
        [256, 256],
        [512, 512],
        [1024, 1024],
        
        # Different length combinations (2 requests)
        [128, 512],
        [256, 1024],
        [512, 2048],
        
        # More complex batches (3-4 requests)
        [128, 256, 512],
        [256, 512, 1024],
        [128, 256, 512, 1024],
        
        # Extreme cases
        [128, 2048],
        [128, 128, 128, 2048],
    ]
    return test_cases


def calculate_equivalent_length(prefill_lengths: List[int]) -> int:
    """Calculate the equivalent single prefill length according to the paper's formula."""
    sum_of_squares = sum(length**2 for length in prefill_lengths)
    return int(np.sqrt(sum_of_squares))


def profile_attention_batch(
    model_wrapper: AttentionWrapper,
    prefill_lengths: List[int],
) -> Dict:
    """Profile a batch of prefill requests with different lengths."""
    results = []
    
    # Create batch with multiple requests (one by one since AttentionWrapper supports batch_size=1 for prefill)
    for prefill_length in prefill_lengths:
        attention_input = AttentionInput(
            prefill_chunk_size=prefill_length,
            kv_cache_size=0,  # No KV cache for fresh prefill
            batch_size=1,     # Always 1 for prefill in current implementation
            is_prefill=True,
        )
        result = model_wrapper.profile(attention_input)
        results.append(result)
    
    # Sum up the time for all requests in batch
    total_time = sum(result["time_stats.mean"] for result in results)
    return {"actual_time": total_time, "prefill_lengths": prefill_lengths}


def profile_equivalent_prefill(
    model_wrapper: AttentionWrapper,
    equivalent_length: int,
) -> Dict:
    """Profile a single prefill with the equivalent length."""
    attention_input = AttentionInput(
        prefill_chunk_size=equivalent_length,
        kv_cache_size=0,
        batch_size=1,
        is_prefill=True,
    )
    result = model_wrapper.profile(attention_input)
    return {"estimated_time": result["time_stats.mean"], "equivalent_length": equivalent_length}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and configuration
    model_config = ModelConfig.from_model_name(args.model)
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.num_tensor_parallel,
        pipeline_parallel_size=1,
    )
    dtype = torch.float16
    
    # Calculate max_num_blocks for the model
    max_num_blocks = get_max_num_blocks(
        model_config,
        parallel_config,
        args.block_size,
        dtype,
    )
    
    # Initialize attention wrapper
    model_wrapper = AttentionWrapper(
        model_config,
        parallel_config,
        max_num_blocks,
        args.max_model_len,
        args.block_size,
        args.attention_backend,
        dtype,
    )
    
    # Generate test cases
    test_cases = generate_test_cases()
    results = []
    
    # Run the experiments
    for prefill_lengths in tqdm(test_cases):
        # Skip if any length exceeds max_model_len
        if any(length > args.max_model_len for length in prefill_lengths):
            continue
            
        # Calculate equivalent length
        equivalent_length = calculate_equivalent_length(prefill_lengths)
        if equivalent_length > args.max_model_len:
            continue
            
        # Measure actual batch execution time
        actual_result = profile_attention_batch(model_wrapper, prefill_lengths)
        
        # Measure estimated time using equivalent length
        estimated_result = profile_equivalent_prefill(model_wrapper, equivalent_length)
        
        # Combine results
        result = {
            "prefill_lengths": prefill_lengths,
            "equivalent_length": equivalent_length,
            "actual_time": actual_result["actual_time"],
            "estimated_time": estimated_result["estimated_time"],
            "error_percent": (estimated_result["estimated_time"] - actual_result["actual_time"]) 
                            / actual_result["actual_time"] * 100
        }
        results.append(result)
    
    # Convert to DataFrame and save results
    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"{args.output_dir}/prefill_estimation_results_{timestamp}.csv", index=False)
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df["actual_time"], df["estimated_time"])
    
    # Add perfect estimation line
    min_time = min(df["actual_time"].min(), df["estimated_time"].min())
    max_time = max(df["actual_time"].max(), df["estimated_time"].max())
    plt.plot([min_time, max_time], [min_time, max_time], 'r--')
    
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Estimated Execution Time (ms)")
    plt.title(f"Prefill Estimation Accuracy - {args.model}")
    plt.savefig(f"{args.output_dir}/prefill_estimation_plot_{timestamp}.png")
    
    # Print summary statistics
    print(f"Average Error: {df['error_percent'].mean():.2f}%")
    print(f"Max Error: {df['error_percent'].abs().max():.2f}%")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()