from math import ceil
from typing import List
import numpy as np
import torch

from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy
from sarathi.model_executor.attention import get_attention_wrapper


class MixedBatchAttentionInput:
    """Custom input class that supports mixed-length batches for prefill."""
    def __init__(self, prefill_lengths: List[int]):
        self.prefill_lengths = prefill_lengths
        self.batch_size = len(prefill_lengths)
        self.is_prefill = True
        
    def is_valid(self, max_seq_len: int):
        return all(length > 0 and length <= max_seq_len for length in self.prefill_lengths)
    
    def is_under_memory_limit(self, max_num_tokens: int):
        return sum(self.prefill_lengths) <= max_num_tokens


class MixedBatchAttentionWrapper(AttentionWrapper):
    """Extended AttentionWrapper that supports mixed-length batches for prefill."""
    
    def _get_mixed_batch_input_tensors(self, mixed_input: MixedBatchAttentionInput):
        """Create input tensors for a mixed-length batch."""
        # Calculate total number of tokens across all sequences
        total_tokens = sum(mixed_input.prefill_lengths)
        
        # Create concatenated query, key, value tensors
        query = torch.randn(
            total_tokens,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        
        # Create sequence metadata for each sequence in the batch
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for prefill_length in mixed_input.prefill_lengths:
            num_blocks = ceil(prefill_length / self._block_size)
            seq_metadata = SequenceMetadataProxy(
                is_prompt=True,
                total_len=prefill_length,
                processed_len=0,  # Fresh prefill
                block_table=np.random.default_rng()
                .integers(low=0, high=self.max_num_blocks - 1, size=num_blocks)
                .tolist(),
            )
            seq_metadata_list.append(seq_metadata)
        
        return seq_metadata_list, query, key, value, self.kv_cache
    
    def _extract_mean_time(self, time_stats_dict):
        """Extract mean timing from time_stats dictionary."""
        if not time_stats_dict:
            return 0.0
        
        # time_stats is a nested dict: {operation_name: {stat_name: value}}
        # We want to get the total time across all operations
        total_time = 0.0
        operation_count = 0
        
        for operation_name, stats in time_stats_dict.items():
            if isinstance(stats, dict) and 'mean' in stats:
                total_time += stats['mean']
                operation_count += 1
        
        return total_time if operation_count > 0 else 0.0
    
    @torch.inference_mode()
    def profile_mixed_batch(self, mixed_input: MixedBatchAttentionInput, warmup_steps: int = 2, active_steps: int = 5):
        """Profile a mixed-length batch."""
        try:
            assert mixed_input.is_valid(self._max_model_len)
            assert mixed_input.is_under_memory_limit(self.max_num_blocks * self._block_size)
            
            seq_metadata_list, query, key, value, kv_cache = self._get_mixed_batch_input_tensors(mixed_input)
            get_attention_wrapper().begin_forward(seq_metadata_list)
            
            # Warmup
            for _ in range(warmup_steps):
                get_attention_wrapper().forward(query, key, value, kv_cache)
            torch.cuda.synchronize()
            
            # Clear stats and measure
            self.time_stats_store.clear_stats()
            
            for _ in range(active_steps):
                get_attention_wrapper().forward(query, key, value, kv_cache)
            torch.cuda.synchronize()
            
            get_attention_wrapper().end_forward()
            
            time_stats = self.time_stats_store.get_stats()
            mean_time = self._extract_mean_time(time_stats)
            
            return {
                "time_stats": time_stats,
                "mean_time": mean_time,
                "prefill_lengths": mixed_input.prefill_lengths,
                "batch_size": mixed_input.batch_size,
                "total_tokens": sum(mixed_input.prefill_lengths),
                "attention_backend": self._attention_backend,
            }
            
        except Exception as e:
            print(f"Error in profile_mixed_batch: {e}")
            return {
                "time_stats": {},
                "mean_time": 0.0,
                "prefill_lengths": mixed_input.prefill_lengths,
                "batch_size": mixed_input.batch_size,
                "total_tokens": sum(mixed_input.prefill_lengths),
                "attention_backend": self._attention_backend,
            }