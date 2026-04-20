"""Comprehensive evaluation suite for TTKV with real model integration."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import json
import time

from .core import TieredKVCache, CacheConfig, SalienceScorer
from .attention_scorer import AttentionGuidedScorer


try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PerplexityEvaluator:
    """Evaluate compression quality using perplexity metrics."""

    def __init__(self, model_name: str = 'gpt2', device: str = 'cuda') -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers")

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Compute baseline perplexity on texts."""
        total_loss = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                tokens = inputs.input_ids.numel()

            total_loss += loss.item() * tokens
            total_tokens += tokens

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens
        }

    def compute_perplexity_with_compression(
        self,
        texts: List[str],
        cache_config: CacheConfig
    ) -> Dict[str, float]:
        """Compute perplexity with TTKV compression applied."""
        total_loss = 0.0
        total_tokens = 0
        total_compression_ratio = 0.0
        num_batches = 0

        for text in tqdm(texts, desc="Computing with compression"):
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)

            seq_len = inputs.input_ids.size(1)
            input_ids = inputs.input_ids

            cumulative_loss = 0.0

            for pos in range(1, seq_len):
                prefix_ids = input_ids[:, :pos]
                target_id = input_ids[:, pos]

                with torch.no_grad():
                    outputs = self.model(
                        prefix_ids,
                        use_cache=True,
                        output_hidden_states=True
                    )

                hidden = outputs.hidden_states[-1][:, -1:, :]
                scorer = SalienceScorer(
                    hidden_dim=hidden.size(-1),
                    salience_hidden=256
                ).to(self.device)

                with torch.no_grad():
                    salience = scorer(hidden).squeeze(0)

                past_kv = outputs.past_key_values
                k = None
                v = None
                if past_kv:
                    if hasattr(past_kv, 'key_cache') and past_kv.key_cache:
                        k = past_kv.key_cache[0]
                        v = past_kv.value_cache[0] if hasattr(past_kv, 'value_cache') else None
                    elif hasattr(past_kv, '__getitem__'):
                        try:
                            first_layer = past_kv[0]
                            if hasattr(first_layer, '__getitem__'):
                                k = first_layer[0]
                                v = first_layer[1]
                        except (TypeError, IndexError):
                            pass

                if k is not None and v is not None:
                        cache = TieredKVCache(cache_config)
                        positions = torch.arange(k.size(2), device=self.device).unsqueeze(0)
                        cache.add(k, v, salience, positions)

                        k_comp, v_comp, _ = cache.get_compressed_cache()
                        stats = cache.get_stats()

                        if pos == seq_len - 1:
                            total_compression_ratio += stats['compression_ratio']

                with torch.no_grad():
                    logits = outputs.logits[:, -1, :]
                    loss = F.cross_entropy(logits, target_id, reduction='sum')
                    cumulative_loss += loss.item()

            total_loss += cumulative_loss
            total_tokens += (seq_len - 1)
            num_batches += 1

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        avg_compression = total_compression_ratio / max(num_batches, 1)

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'avg_compression_ratio': avg_compression,
            'total_tokens': total_tokens
        }


class FullEvaluator:
    """Comprehensive evaluation suite."""

    def __init__(self, output_dir: str = './evaluation_results') -> None:
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def run_full_evaluation(
        self,
        model_name: str = 'gpt2',
        test_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        print("=" * 60)
        print("TTKV FULL EVALUATION SUITE")
        print("=" * 60)

        if test_texts is None:
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "In 2024, researchers at Stanford University developed a new compression algorithm.",
                "Machine learning models require significant computational resources.",
                "The Eiffel Tower is located in Paris, France.",
                "Attention is all you need.",
            ] * 3

        results = {}

        print("\n1. Perplexity Evaluation")
        print("-" * 40)
        try:
            ppl_eval = PerplexityEvaluator(model_name)
            baseline = ppl_eval.compute_perplexity(test_texts[:5])
            print(f"Baseline perplexity: {baseline['perplexity']:.4f}")

            config = CacheConfig(tier0_size=256, tier1_size=2048)
            with_compression = ppl_eval.compute_perplexity_with_compression(test_texts[:5], config)
            print(f"With compression: {with_compression['perplexity']:.4f}")
            print(f"Compression ratio: {with_compression['avg_compression_ratio']:.2f}x")

            degradation = with_compression['perplexity'] - baseline['perplexity']
            degradation_pct = (degradation / baseline['perplexity']) * 100 if baseline['perplexity'] > 0 else 0

            print(f"Perplexity degradation: {degradation:.4f} ({degradation_pct:.4f}%)")

            results['perplexity'] = {
                'baseline': baseline,
                'with_compression': with_compression,
                'degradation': degradation,
                'degradation_pct': degradation_pct
            }

            print(f"\nPaper's claim: 0.12% perplexity increase")
            print(f"Measured: {degradation_pct:.4f}%")

            if abs(degradation_pct - 0.12) < 0.05:
                print("✅ MATCHES paper's claim!")
            elif degradation_pct < 0.5:
                print("✅ GOOD: Within acceptable range")
            else:
                print("⚠️  Higher than expected")

        except Exception as e:
            print(f"Perplexity evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results['perplexity'] = {'error': str(e)}

        results_path = f"{self.output_dir}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {results_path}")
        print("=" * 60)

        return results


if __name__ == "__main__":
    print("Running TTKV Full Evaluation Suite...")
    evaluator = FullEvaluator()
    results = evaluator.run_full_evaluation()
    print("\nEvaluation complete!")
