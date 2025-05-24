import argparse
import datetime
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import random

# IMPORTANT: Apply monkey patch BEFORE importing any sarathi modules
import sarathi.metrics.cuda_timer
from vidur.profiling.common.cuda_timer import CudaTimer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

# Initialize dummy MetricsStore
from sarathi.metrics.metrics_store import MetricsStore

class DummyReplicaConfig:
    def __init__(self):
        self.replica_id = 0
        self.output_dir = "/tmp"

class DummyMetricsConfig:
    def __init__(self):
        self.write_metrics = False
        self.wandb_project = None
        self.wandb_group = None

class DummySarathiModelConfig:
    def __init__(self):
        pass
    def get_total_num_layers(self):
        return 32

try:
    MetricsStore.get_or_create_instance(
        DummyReplicaConfig(), DummySarathiModelConfig(), DummyMetricsConfig()
    )
except Exception:
    pass

from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import AttentionBackend

from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.utils import get_max_num_blocks
from vidur.profiling.prefill_calculation.mixed_batch_attention_wrapper import MixedBatchAttentionWrapper, MixedBatchAttentionInput


def parse_args():
    parser = argparse.ArgumentParser(description="Prefill Estimation Accuracy Test V2")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--attention_backend", default=AttentionBackend.FLASHINFER, choices=[e.value for e in AttentionBackend])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--measurement_runs", type=int, default=10, help="Number of measurement runs")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--num_test_cases", type=int, default=150, help="Number of test cases to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible test cases")
    return parser.parse_args()


def generate_comprehensive_test_cases(max_model_len: int, num_cases: int = 150, seed: int = 42) -> List[List[int]]:
    """Generate comprehensive test cases for mixed-length batches."""
    random.seed(seed)
    np.random.seed(seed)
    
    test_cases = []
    
    # 1. Basic single prefill cases for baseline
    basic_lengths = [64, 128, 256, 512, 1024, 2048, 3072]
    for length in basic_lengths:
        if length <= max_model_len:
            test_cases.append([length])
    
    # 2. Equal length batches (batch sizes 2-6)
    equal_lengths = [64, 128, 256, 512, 1024, 1536, 2048]
    for batch_size in [2, 3, 4, 5, 6]:
        for length in equal_lengths:
            if length * batch_size <= max_model_len * 0.8:  # Conservative memory limit
                test_cases.append([length] * batch_size)
    
    # 3. Arithmetic progression sequences
    for batch_size in [3, 4, 5]:
        for start in [64, 128, 256]:
            for step in [64, 128, 256]:
                sequence = [start + i * step for i in range(batch_size)]
                if max(sequence) <= max_model_len and sum(sequence) <= max_model_len * 0.8:
                    test_cases.append(sequence)
    
    # 4. Geometric progression sequences
    for batch_size in [3, 4, 5]:
        for start in [32, 64, 128]:
            for ratio in [1.5, 2.0, 2.5]:
                sequence = [int(start * (ratio ** i)) for i in range(batch_size)]
                if max(sequence) <= max_model_len and sum(sequence) <= max_model_len * 0.8:
                    test_cases.append(sequence)
    
    # 5. Extreme combinations (very short + very long)
    short_lengths = [32, 64, 96, 128]
    long_lengths = [1536, 2048, 2560, 3072]
    for short in short_lengths:
        for long in long_lengths:
            if long <= max_model_len and short + long <= max_model_len * 0.8:
                # One short + one long
                test_cases.append([short, long])
                # Multiple short + one long
                if 3 * short + long <= max_model_len * 0.8:
                    test_cases.append([short, short, short, long])
                if 4 * short + long <= max_model_len * 0.8:
                    test_cases.append([short, short, short, short, long])
    
    # 6. Random combinations with different patterns
    for _ in range(40):  # Generate 40 random cases
        batch_size = random.randint(2, 6)
        
        # Random uniform distribution
        max_length = min(max_model_len, 2048)
        lengths = [random.randint(64, max_length) for _ in range(batch_size)]
        if sum(lengths) <= max_model_len * 0.8:
            test_cases.append(sorted(lengths))
    
    # 7. Fibonacci-like sequences
    for start1, start2 in [(32, 64), (64, 128), (96, 160)]:
        fib_seq = [start1, start2]
        while len(fib_seq) < 6:
            next_val = fib_seq[-1] + fib_seq[-2]
            if next_val > max_model_len:
                break
            fib_seq.append(next_val)
        
        # Generate subsequences of different lengths
        for length in range(2, min(6, len(fib_seq) + 1)):
            subseq = fib_seq[:length]
            if sum(subseq) <= max_model_len * 0.8:
                test_cases.append(subseq)
    
    # 8. Power of 2 sequences
    powers = [32, 64, 128, 256, 512, 1024, 2048]
    for batch_size in [2, 3, 4, 5]:
        for start_idx in range(len(powers) - batch_size + 1):
            sequence = powers[start_idx:start_idx + batch_size]
            if sum(sequence) <= max_model_len * 0.8:
                test_cases.append(sequence)
    
    # 9. Mixed distribution patterns
    common_lengths = [128, 256, 512, 1024]
    for _ in range(30):  # 30 mixed cases
        batch_size = random.randint(2, 5)
        # Create bimodal distribution
        if random.random() < 0.5:
            # Mostly small with one large
            lengths = [random.choice(common_lengths[:2]) for _ in range(batch_size - 1)]
            lengths.append(random.choice(common_lengths[2:]))
        else:
            # Mostly large with one small
            lengths = [random.choice(common_lengths[2:]) for _ in range(batch_size - 1)]
            lengths.append(random.choice(common_lengths[:2]))
        
        if sum(lengths) <= max_model_len * 0.8:
            test_cases.append(sorted(lengths))
    
    # 10. Edge cases
    # Very small lengths
    for batch_size in [2, 3, 4, 5, 6, 7, 8]:
        small_lengths = [32] * batch_size
        if sum(small_lengths) <= max_model_len * 0.8:
            test_cases.append(small_lengths)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_test_cases = []
    for case in test_cases:
        case_tuple = tuple(sorted(case))
        if case_tuple not in seen:
            seen.add(case_tuple)
            unique_test_cases.append(case)
    
    # Shuffle to randomize order
    random.shuffle(unique_test_cases)
    
    # Truncate or extend to desired number
    if len(unique_test_cases) > num_cases:
        unique_test_cases = unique_test_cases[:num_cases]
    elif len(unique_test_cases) < num_cases:
        # If we need more cases, generate additional random ones
        while len(unique_test_cases) < num_cases:
            batch_size = random.randint(2, 5)
            max_length = min(max_model_len, 2048)
            lengths = sorted([random.randint(64, max_length) for _ in range(batch_size)])
            if sum(lengths) <= max_model_len * 0.8:
                case_tuple = tuple(lengths)
                if case_tuple not in seen:
                    seen.add(case_tuple)
                    unique_test_cases.append(lengths)
    
    return unique_test_cases


def calculate_equivalent_length(prefill_lengths: List[int]) -> int:
    """Calculate equivalent length using vidur's formula: sqrt(sum(p_i^2))"""
    sum_of_squares = sum(length**2 for length in prefill_lengths)
    return int(np.sqrt(sum_of_squares))


def extract_mean_time_from_stats(time_stats):
    """Extract mean time from time_stats dictionary."""
    if not time_stats:
        return 0.0
    
    # time_stats is a nested dict: {operation_name: {stat_name: value}}
    total_time = 0.0
    operation_count = 0
    
    for operation_name, stats in time_stats.items():
        if isinstance(stats, dict) and 'mean' in stats:
            total_time += stats['mean']
            operation_count += 1
    
    return total_time if operation_count > 0 else 0.0


def profile_single_prefill(wrapper: MixedBatchAttentionWrapper, length: int, warmup: int, runs: int, debug: bool = False) -> float:
    """Profile a single prefill request with improved warmup strategy."""
    attention_input = AttentionInput(
        prefill_chunk_size=length,
        kv_cache_size=0,
        batch_size=1,
        is_prefill=True,
    )
    
    # Perform warmup runs (don't collect timing)
    for _ in range(warmup):
        try:
            wrapper.profile(attention_input)
        except Exception as e:
            if debug:
                print(f"Warmup error for single prefill length {length}: {e}")
            return 0.0
    
    # Perform measurement runs
    times = []
    for run_idx in range(runs):
        try:
            result = wrapper.profile(attention_input)
            time_stats = result.get("time_stats", {})
            mean_time = extract_mean_time_from_stats(time_stats)
            times.append(mean_time)
            if debug:
                print(f"  Single prefill {length}, run {run_idx+1}: {mean_time:.3f}ms")
        except Exception as e:
            if debug:
                print(f"Error profiling single prefill length {length}, run {run_idx}: {e}")
            return 0.0
    
    return np.mean(times) if times else 0.0


def profile_mixed_batch(wrapper: MixedBatchAttentionWrapper, prefill_lengths: List[int], warmup: int, runs: int, debug: bool = False) -> float:
    """Profile a true mixed-length batch with improved warmup strategy."""
    mixed_input = MixedBatchAttentionInput(prefill_lengths)
    
    # Perform warmup runs
    for _ in range(warmup):
        try:
            wrapper.profile_mixed_batch(mixed_input, warmup_steps=2, active_steps=1)
        except Exception as e:
            if debug:
                print(f"Warmup error for mixed batch {prefill_lengths}: {e}")
            return 0.0
    
    # Perform measurement runs
    times = []
    for run_idx in range(runs):
        try:
            # For measurement, use minimal internal warmup since we already warmed up
            result = wrapper.profile_mixed_batch(mixed_input, warmup_steps=1, active_steps=3)
            mean_time = result.get("mean_time", 0.0)
            times.append(mean_time)
            if debug:
                print(f"  Mixed batch {prefill_lengths}, run {run_idx+1}: {mean_time:.3f}ms")
        except Exception as e:
            if debug:
                print(f"Error profiling mixed batch {prefill_lengths}, run {run_idx}: {e}")
            return 0.0
    
    return np.mean(times) if times else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    model_config = ModelConfig.from_model_name(args.model)
    parallel_config = ParallelConfig(tensor_parallel_size=args.num_tensor_parallel, pipeline_parallel_size=1)
    dtype = torch.float16
    max_num_blocks = get_max_num_blocks(model_config, parallel_config, args.block_size, dtype)
    
    print(f"Initializing MixedBatchAttentionWrapper for {args.model}")
    print(f"Max num blocks: {max_num_blocks}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Block size: {args.block_size}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Measurement runs: {args.measurement_runs}")
    
    # Initialize wrapper
    try:
        wrapper = MixedBatchAttentionWrapper(
            model_config, parallel_config, max_num_blocks, 
            args.max_model_len, args.block_size, args.attention_backend, dtype
        )
        print("MixedBatchAttentionWrapper initialized successfully")
    except Exception as e:
        print(f"Failed to initialize wrapper: {e}")
        return
    
    # Generate comprehensive test cases
    test_cases = generate_comprehensive_test_cases(args.max_model_len, args.num_test_cases, args.seed)
    results = []
    
    print(f"Generated {len(test_cases)} test cases")
    print(f"Running experiments...")
    
    for i, prefill_lengths in enumerate(tqdm(test_cases)):
        # Skip if invalid
        if any(length > args.max_model_len for length in prefill_lengths):
            if args.debug:
                print(f"Skipping case {i+1}: lengths exceed max_model_len")
            continue
        
        equivalent_length = calculate_equivalent_length(prefill_lengths)
        if equivalent_length > args.max_model_len:
            if args.debug:
                print(f"Skipping case {i+1}: equivalent length {equivalent_length} exceeds max_model_len")
            continue
        
        total_tokens = sum(prefill_lengths)
        max_tokens = max_num_blocks * args.block_size
        if total_tokens > max_tokens:
            if args.debug:
                print(f"Skipping case {i+1}: total tokens {total_tokens} exceed limit {max_tokens}")
            continue
            
        if args.debug and i < 5:  # Only print details for first few cases
            print(f"Test case {i+1}: {prefill_lengths} -> equivalent: {equivalent_length}")
        
        try:
            # Test single prefill first (for debugging and baseline)
            if len(prefill_lengths) == 1:
                single_time = profile_single_prefill(
                    wrapper, prefill_lengths[0], args.warmup_runs, args.measurement_runs, args.debug and i < 5
                )
                
                if single_time > 0:
                    result = {
                        "test_case": i+1,
                        "prefill_lengths": str(prefill_lengths),
                        "batch_size": len(prefill_lengths),
                        "total_tokens": total_tokens,
                        "equivalent_length": equivalent_length,
                        "actual_time_ms": single_time,
                        "estimated_time_ms": single_time,  # Same for single prefill
                        "error_percent": 0.0,
                        "abs_error_percent": 0.0,
                        "test_type": "single_prefill"
                    }
                    results.append(result)
                continue
            
            # Method 1: True mixed batch (ACTUAL)
            actual_time = profile_mixed_batch(
                wrapper, prefill_lengths, args.warmup_runs, args.measurement_runs, args.debug and i < 5
            )
            
            # Method 2: Equivalent single prefill (VIDUR ESTIMATION)
            estimated_time = profile_single_prefill(
                wrapper, equivalent_length, args.warmup_runs, args.measurement_runs, args.debug and i < 5
            )
            
            if actual_time > 0 and estimated_time > 0:
                error_percent = (estimated_time - actual_time) / actual_time * 100
                
                result = {
                    "test_case": i+1,
                    "prefill_lengths": str(prefill_lengths),
                    "batch_size": len(prefill_lengths),
                    "total_tokens": total_tokens,
                    "equivalent_length": equivalent_length,
                    "actual_time_ms": actual_time,
                    "estimated_time_ms": estimated_time,
                    "error_percent": error_percent,
                    "abs_error_percent": abs(error_percent),
                    "test_type": "mixed_batch"
                }
                results.append(result)
                
                if args.debug and i < 5:
                    print(f"  Actual: {actual_time:.3f}ms, Estimated: {estimated_time:.3f}ms, Error: {error_percent:.1f}%")
            else:
                if args.debug and i < 5:
                    print(f"  Skipped due to measurement failure (actual: {actual_time}, estimated: {estimated_time})")
                
        except Exception as e:
            if args.debug and i < 5:
                print(f"Error in test case {i+1}: {e}")
            continue
    
    if not results:
        print("No successful test cases!")
        return
    
    # Save and analyze results
    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save CSV
    csv_path = f"{args.output_dir}/comprehensive_estimation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Filter results by type
    single_df = df[df['test_type'] == 'single_prefill']
    mixed_df = df[df['test_type'] == 'mixed_batch']
    
    print(f"\nResults Summary:")
    print(f"  Total test cases run: {len(df)}")
    print(f"  Single prefill cases: {len(single_df)}")
    print(f"  Mixed batch cases: {len(mixed_df)}")
    
    if len(mixed_df) == 0:
        print("No mixed batch test cases succeeded!")
        print(f"Results saved to: {csv_path}")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # 1. Actual vs Estimated scatter
    ax = axes[0, 0]
    ax.scatter(mixed_df["actual_time_ms"], mixed_df["estimated_time_ms"], alpha=0.6, s=30)
    min_time = min(mixed_df["actual_time_ms"].min(), mixed_df["estimated_time_ms"].min())
    max_time = max(mixed_df["actual_time_ms"].max(), mixed_df["estimated_time_ms"].max())
    ax.plot([min_time, max_time], [min_time, max_time], 'r--', linewidth=2, label='Perfect Estimation')
    ax.set_xlabel("Actual Execution Time (ms)")
    ax.set_ylabel("Estimated Execution Time (ms)")
    ax.set_title("Actual vs Estimated Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(mixed_df["error_percent"], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(mixed_df["error_percent"].mean(), color='red', linestyle='--', 
               label=f'Mean: {mixed_df["error_percent"].mean():.1f}%')
    ax.axvline(mixed_df["error_percent"].median(), color='blue', linestyle='--', 
               label=f'Median: {mixed_df["error_percent"].median():.1f}%')
    ax.set_xlabel("Error Percentage (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Estimation Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error vs batch size
    ax = axes[0, 2]
    for batch_size in sorted(mixed_df["batch_size"].unique()):
        subset = mixed_df[mixed_df["batch_size"] == batch_size]
        ax.scatter([batch_size] * len(subset), subset["abs_error_percent"], alpha=0.6, s=30, label=f'Size {batch_size}')
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("Error vs Batch Size")
    ax.grid(True, alpha=0.3)
    
    # 4. Error vs total tokens
    ax = axes[1, 0]
    ax.scatter(mixed_df["total_tokens"], mixed_df["abs_error_percent"], alpha=0.6, s=30)
    ax.set_xlabel("Total Tokens in Batch")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("Error vs Total Tokens")
    ax.grid(True, alpha=0.3)
    
    # 5. Error vs equivalent length
    ax = axes[1, 1]
    ax.scatter(mixed_df["equivalent_length"], mixed_df["abs_error_percent"], alpha=0.6, s=30)
    ax.set_xlabel("Equivalent Length")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("Error vs Equivalent Length")
    ax.grid(True, alpha=0.3)
    
    # 6. Box plot of errors by batch size
    ax = axes[1, 2]
    batch_sizes = sorted(mixed_df["batch_size"].unique())
    error_by_batch = [mixed_df[mixed_df["batch_size"] == bs]["error_percent"].values for bs in batch_sizes]
    ax.boxplot(error_by_batch, labels=batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Error Percentage (%)")
    ax.set_title("Error Distribution by Batch Size")
    ax.grid(True, alpha=0.3)
    
    # 7. Cumulative error distribution
    ax = axes[2, 0]
    sorted_errors = np.sort(mixed_df["abs_error_percent"])
    y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, y_vals)
    ax.set_xlabel("Absolute Error (%)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Error Distribution")
    ax.grid(True, alpha=0.3)
    
    # 8. Timing comparison for subset
    ax = axes[2, 1]
    subset_idx = np.linspace(0, len(mixed_df)-1, min(20, len(mixed_df)), dtype=int)
    subset_df = mixed_df.iloc[subset_idx]
    x_pos = np.arange(len(subset_df))
    width = 0.35
    ax.bar(x_pos - width/2, subset_df["actual_time_ms"], width, label='Actual (Mixed Batch)', alpha=0.7)
    ax.bar(x_pos + width/2, subset_df["estimated_time_ms"], width, label='Estimated (Equivalent)', alpha=0.7)
    ax.set_xlabel("Sample Test Cases")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Timing Comparison (Sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Error vs length variance
    ax = axes[2, 2]
    length_vars = []
    for _, row in mixed_df.iterrows():
        lengths = eval(row["prefill_lengths"])  # Convert string back to list
        length_vars.append(np.var(lengths))
    ax.scatter(length_vars, mixed_df["abs_error_percent"], alpha=0.6, s=30)
    ax.set_xlabel("Length Variance")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("Error vs Length Variance")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{args.output_dir}/comprehensive_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE VIDUR PREFILL ESTIMATION ACCURACY ANALYSIS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Total test cases: {len(df)}")
    print(f"Mixed batch cases: {len(mixed_df)}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Warmup strategy: {args.warmup_runs} warmup + {args.measurement_runs} measurement runs")
    print("-"*80)
    
    if len(mixed_df) > 0:
        print("ERROR STATISTICS (Mixed Batch Only):")
        print(f"  Mean Error: {mixed_df['error_percent'].mean():.2f}%")
        print(f"  Median Error: {mixed_df['error_percent'].median():.2f}%")
        print(f"  Std Deviation: {mixed_df['error_percent'].std():.2f}%")
        print(f"  Min Error: {mixed_df['error_percent'].min():.2f}%")
        print(f"  Max Error: {mixed_df['error_percent'].max():.2f}%")
        print(f"  Mean Absolute Error: {mixed_df['abs_error_percent'].mean():.2f}%")
        print(f"  95th Percentile Abs Error: {np.percentile(mixed_df['abs_error_percent'], 95):.2f}%")
        
        print("\nACCURACY THRESHOLDS:")
        for threshold in [5, 10, 15, 20, 25, 30, 50]:
            within_threshold = (mixed_df['abs_error_percent'] <= threshold).sum()
            percentage = within_threshold / len(mixed_df) * 100
            print(f"  Within {threshold}%: {within_threshold}/{len(mixed_df)} ({percentage:.1f}%)")
        
        print("\nBATCH SIZE ANALYSIS:")
        for batch_size in sorted(mixed_df['batch_size'].unique()):
            subset = mixed_df[mixed_df['batch_size'] == batch_size]
            mean_error = subset['abs_error_percent'].mean()
            median_error = subset['abs_error_percent'].median()
            print(f"  Batch size {batch_size}: {len(subset)} cases, mean error: {mean_error:.1f}%, median: {median_error:.1f}%")
    
    print(f"\nOUTPUT FILES:")
    print(f"  CSV: {csv_path}")
    print(f"  Plot: {plot_path}")
    print("="*80)
    
    # Additional insights
    if len(mixed_df) > 0:
        best_case = mixed_df.loc[mixed_df['abs_error_percent'].idxmin()]
        worst_case = mixed_df.loc[mixed_df['abs_error_percent'].idxmax()]
        
        print(f"\nBEST CASE (lowest error):")
        print(f"  Lengths: {best_case['prefill_lengths']}")
        print(f"  Error: {best_case['error_percent']:.1f}%")
        print(f"  Actual: {best_case['actual_time_ms']:.1f}ms, Estimated: {best_case['estimated_time_ms']:.1f}ms")
        
        print(f"\nWORST CASE (highest error):")
        print(f"  Lengths: {worst_case['prefill_lengths']}")
        print(f"  Error: {worst_case['error_percent']:.1f}%")
        print(f"  Actual: {worst_case['actual_time_ms']:.1f}ms, Estimated: {worst_case['estimated_time_ms']:.1f}ms")


if __name__ == "__main__":
    main()