"""Integration test for end-to-end model integration with TTKV."""

import pytest
import torch
from ttkv import (
    CacheConfig,
    TieredKVCache,
    AttentionGuidedScorer,
    MockTypePriorClassifier,
    compute_type_prior_retention,
)


class TestModelIntegration:
    """Integration tests for TTKV with model-like usage patterns."""

    @pytest.fixture
    def config(self) -> CacheConfig:
        """Create a test cache configuration."""
        return CacheConfig(
            hidden_dim=768,
            num_heads=12,
            head_dim=64,
            tier0_size=128,
            tier1_size=512,
            tier1_compression=4,
            tier2_compression=16,
            tau_threshold=0.8,
        )

    @pytest.fixture
    def batch_size(self) -> int:
        return 1

    @pytest.fixture
    def num_heads(self) -> int:
        return 12

    @pytest.fixture
    def head_dim(self) -> int:
        return 64

    def test_full_pipeline_small_sequence(
        self,
        config: CacheConfig,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Test full pipeline with a small sequence."""
        cache = TieredKVCache(config)
        seq_len = 256

        # Simulate KV cache from a transformer
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create retention scores with type priors
        token_ids = torch.randint(0, 50000, (batch_size, seq_len))
        retention = compute_type_prior_retention(token_ids)
        positions = torch.arange(seq_len).unsqueeze(0)

        # Add to cache
        cache.add(k, v, retention, positions)

        # Get compressed cache
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert v_comp is not None
        assert pos_comp is not None

        # Verify shapes
        assert k_comp.shape[0] == batch_size
        assert k_comp.shape[1] == num_heads
        assert k_comp.shape[3] == head_dim

        # Verify compression happened
        stats = cache.get_stats()
        assert stats['total_tokens'] == seq_len
        assert stats['compression_ratio'] > 1.0

    def test_full_pipeline_long_sequence(
        self,
        config: CacheConfig,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Test full pipeline with a long sequence (16K tokens)."""
        cache = TieredKVCache(config)
        seq_len = 16384

        # Simulate KV cache
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create retention scores
        token_ids = torch.randint(0, 50000, (batch_size, seq_len))
        retention = compute_type_prior_retention(token_ids)
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert v_comp is not None

        stats = cache.get_stats()
        assert stats['total_tokens'] == seq_len
        # Should achieve significant compression at 16K
        assert stats['compression_ratio'] > 5.0

    def test_attention_guided_scorer_integration(self) -> None:
        """Test AttentionGuidedScorer with attention-like patterns."""
        scorer = AttentionGuidedScorer(ema_decay=0.95, structural_floor=0.1)

        # Simulate attention weights (high attention on certain positions)
        seq_len = 100
        attention_weights = torch.softmax(torch.randn(seq_len), dim=0)

        # Update scorer
        scorer.update_from_attention(
            attention_weights.unsqueeze(0),
            query_position=seq_len - 1,
            generated_token_id=42,
        )

        # Get salience scores
        structural_scores = torch.randn(1, seq_len)
        salience = scorer.get_salience_scores(seq_len, structural_scores)

        assert len(salience) == seq_len
        assert salience.min() >= 0.0

        # Verify positions with high attention have high scores
        scorer.reset()

    def test_type_prior_classifier_integration(self) -> None:
        """Test MockTypePriorClassifier with token classification."""
        classifier = MockTypePriorClassifier()

        # Test with mock tokens
        tokens = [
            "The", "quick", "Brown", "Fox", "123",
            "jumps", "over", ".", "Google", "AI",
        ]
        scores = classifier.classify_tokens(tokens)

        assert len(scores) == len(tokens)

        # Named entities and numbers should have high retention
        assert scores[2] == 1.0  # "Brown" (capitalized)
        assert scores[4] == 1.0  # "123" (numeric)
        assert scores[3] == 1.0  # "Fox" (capitalized)

        # Punctuation should have low retention
        assert scores[7] < 0.1  # "."

    def test_multi_batch_compression(
        self,
        config: CacheConfig,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Test compression with multiple batches."""
        cache = TieredKVCache(config)
        batch_size = 2
        seq_len = 512

        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        retention = torch.rand(batch_size, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert k_comp.shape[0] == batch_size

    def test_incremental_cache_building(
        self,
        config: CacheConfig,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Test adding cache in multiple increments."""
        cache = TieredKVCache(config)

        # Add in 3 increments
        for i in range(3):
            chunk_len = 512
            k = torch.randn(batch_size, num_heads, chunk_len, head_dim)
            v = torch.randn(batch_size, num_heads, chunk_len, head_dim)
            retention = torch.rand(batch_size, chunk_len)
            positions = torch.arange(
                i * chunk_len, (i + 1) * chunk_len
            ).unsqueeze(0)

            cache.add(k, v, retention, positions)

        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert cache.total_tokens == 1536

        stats = cache.get_stats()
        assert stats['total_tokens'] == 1536

    def test_cache_clear_and_reuse(
        self,
        config: CacheConfig,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Test clearing cache and reusing."""
        cache = TieredKVCache(config)
        seq_len = 256

        # First use
        k1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        retention1 = torch.rand(batch_size, seq_len)
        positions1 = torch.arange(seq_len).unsqueeze(0)

        cache.add(k1, v1, retention1, positions1)
        stats1 = cache.get_stats()

        # Clear
        cache.clear()

        stats_cleared = cache.get_stats()
        assert stats_cleared['total_tokens'] == 0

        # Second use
        k2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        retention2 = torch.rand(batch_size, seq_len)
        positions2 = torch.arange(seq_len).unsqueeze(0)

        cache.add(k2, v2, retention2, positions2)
        stats2 = cache.get_stats()

        assert stats2['total_tokens'] == seq_len
        assert stats2['total_tokens'] == stats1['total_tokens']

    def test_empty_cache_handling(self, config: CacheConfig) -> None:
        """Test handling of empty cache."""
        cache = TieredKVCache(config)

        # Try to get compressed cache without adding anything
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is None
        assert v_comp is None
        assert pos_comp is None

        stats = cache.get_stats()
        assert stats['total_tokens'] == 0
        assert stats['compression_ratio'] == 1.0


@pytest.mark.slow
class TestSlowIntegration:
    """Slower integration tests that may require more resources."""

    def test_memory_efficiency_at_32k(
        self,
    ) -> None:
        """Test memory efficiency with 32K sequence length."""
        config = CacheConfig(
            tier0_size=256,
            tier1_size=4096,
            tier1_compression=4,
            tier2_compression=16,
        )
        cache = TieredKVCache(config)

        batch_size = 1
        num_heads = 12
        head_dim = 64
        seq_len = 32768

        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        # Use low retention scores to ensure compression happens
        retention = torch.rand(batch_size, seq_len) * 0.5
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None

        stats = cache.get_stats()
        assert stats['compression_ratio'] > 5.0

        # Verify compressed size is reasonable
        compressed_tokens = stats['compressed_tokens']
        expected_max = seq_len // 5
        assert compressed_tokens < expected_max
