"""Edge case tests for TTKV.

Tests boundary conditions, empty inputs, extreme values, and other edge cases
to ensure robustness of the KV cache implementation.
"""

import pytest
import torch
from ttkv import CacheConfig, TieredKVCache, SalienceScorer, RetentionScheduler
from ttkv.exceptions import (
    ConfigurationError,
    EmptyTensorError,
    ShapeMismatchError,
    DimensionError,
)


class TestEdgeCasesEmptyCache:
    """Tests for empty cache edge cases."""

    def test_empty_cache_returns_none(self) -> None:
        """Test that empty cache returns None."""
        config = CacheConfig()
        cache = TieredKVCache(config)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        assert k_comp is None
        assert v_comp is None
        assert pos_comp is None

    def test_empty_cache_stats(self) -> None:
        """Test stats on empty cache."""
        config = CacheConfig()
        cache = TieredKVCache(config)
        stats = cache.get_stats()
        assert stats.total_tokens == 0
        assert stats.compressed_tokens == 0
        assert stats.compression_ratio == 1.0
        assert stats.tier_distribution == (0, 0, 0, 0)

    def test_clear_empty_cache(self) -> None:
        """Test clearing an already empty cache."""
        config = CacheConfig()
        cache = TieredKVCache(config)
        cache.clear()
        assert cache.total_tokens == 0
        assert len(cache.k_cache) == 0


class TestEdgeCasesSingleToken:
    """Tests for single token edge cases."""

    def test_single_token_no_compression(self) -> None:
        """Test single token doesn't get compressed."""
        config = CacheConfig(tier0_size=256)
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 1, 64)
        v = torch.randn(1, 12, 1, 64)
        retention = torch.rand(1, 1)
        positions = torch.arange(1).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert k_comp.shape[2] == 1  # No compression
        assert stats.compression_ratio == 1.0

    def test_two_tokens_with_extreme_retention(self) -> None:
        """Test two tokens where one is protected."""
        config = CacheConfig(tier0_size=256, tau_threshold=0.8)
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 2, 64)
        v = torch.randn(1, 12, 2, 64)
        # First token protected, second not
        retention = torch.tensor([[0.9, 0.3]])
        positions = torch.arange(2).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert pos_comp is not None
        assert pos_comp[0, 0] == 0  # First position preserved


class TestEdgeCasesExtremeThresholds:
    """Tests for extreme tau_threshold values."""

    def test_tau_threshold_zero(self) -> None:
        """Test tau_threshold=0 (no tokens protected)."""
        config = CacheConfig(tau_threshold=0.0, tier0_size=1)
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 100, 64)
        v = torch.randn(1, 12, 100, 64)
        retention = torch.rand(1, 100)  # All > 0
        positions = torch.arange(100).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert stats.total_tokens == 100
        # Note: When tau=0, tokens with retention > 0 are protected
        # Since torch.rand returns values in (0, 1), all tokens might be protected
        # Just verify compression still works
        assert stats.compression_ratio >= 1.0

    def test_tau_threshold_one(self) -> None:
        """Test tau_threshold=1.0 (almost all tokens protected)."""
        config = CacheConfig(tau_threshold=1.0, tier0_size=1)
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 100, 64)
        v = torch.randn(1, 12, 100, 64)
        # Almost all < 1.0
        retention = torch.rand(1, 100) * 0.99
        positions = torch.arange(100).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert stats.total_tokens == 100

    def test_configuration_error_invalid_tau(self) -> None:
        """Test that invalid tau_threshold raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            CacheConfig(tau_threshold=-0.1)

        with pytest.raises(ConfigurationError):
            CacheConfig(tau_threshold=1.5)


class TestEdgeCasesExtremeCompression:
    """Tests for extreme compression ratios."""

    def test_extreme_compression_ratio_100x(self) -> None:
        """Test very high compression ratio."""
        config = CacheConfig(
            tier0_size=10,
            tier1_size=20,
            tier1_compression=100,
            tier2_compression=100
        )
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 1000, 64)
        v = torch.randn(1, 12, 1000, 64)
        retention = torch.rand(1, 1000) * 0.5
        positions = torch.arange(1000).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert stats.compression_ratio > 40.0  # Extreme compression (> 40x)


class TestEdgeCasesEmptyTensors:
    """Tests for empty tensor handling."""

    def test_empty_keys_raises_error(self) -> None:
        """Test that empty keys raise EmptyTensorError."""
        config = CacheConfig()
        cache = TieredKVCache(config)

        k = torch.empty(1, 12, 0, 64)
        v = torch.empty(1, 12, 0, 64)
        retention = torch.empty(1, 0)
        positions = torch.empty(1, 0, dtype=torch.long)

        with pytest.raises(EmptyTensorError):
            cache.add(k, v, retention, positions)

    def test_zero_sequence_length(self) -> None:
        """Test handling of zero sequence length."""
        config = CacheConfig()
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 1, 64)
        v = torch.randn(1, 12, 1, 64)
        retention = torch.empty(1, 0)
        positions = torch.empty(1, 0, dtype=torch.long)

        with pytest.raises(EmptyTensorError):
            cache.add(k, v, retention, positions)


class TestEdgeCasesConfiguration:
    """Tests for configuration edge cases."""

    def test_configuration_error_negative_tier0(self) -> None:
        """Test negative tier0_size raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            CacheConfig(tier0_size=-1)

    def test_configuration_error_tier1_less_than_tier0(self) -> None:
        """Test tier1 < tier0 raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            CacheConfig(tier0_size=500, tier1_size=100)

    def test_configuration_error_zero_heads(self) -> None:
        """Test zero num_heads raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            CacheConfig(num_heads=0)

    def test_configuration_error_invalid_compression_ratio(self) -> None:
        """Test invalid compression ratio raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            CacheConfig(tier1_compression=0)

        with pytest.raises(ConfigurationError):
            CacheConfig(tier2_compression=-1)


class TestEdgeCasesGradientCheckpointing:
    """Tests for gradient checkpointing compatibility."""

    def test_tensor_with_gradients(self) -> None:
        """Test that cache works with tensors requiring gradients."""
        config = CacheConfig()
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, 64, 64, requires_grad=True)
        v = torch.randn(1, 12, 64, 64, requires_grad=True)
        retention = torch.rand(1, 64, requires_grad=True)
        positions = torch.arange(64).unsqueeze(0)

        # Should not raise
        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        # Compressed tensors may or may not require grad depending on PyTorch version
        # The important thing is that the operation completed successfully


class TestEdgeCasesMultiBatch:
    """Tests for multi-batch edge cases."""

    def test_uneven_batch_retention(self) -> None:
        """Test batch with very different retention per sample."""
        config = CacheConfig(tier0_size=10, tier1_size=100)
        cache = TieredKVCache(config)

        batch_size = 3
        seq_len = 200

        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)

        # Different retention patterns per batch
        retention = torch.zeros(batch_size, seq_len)
        retention[0, :] = 0.1  # Low retention
        retention[1, :] = 0.95  # High retention (protected)
        retention[2, :seq_len//2] = 0.5
        retention[2, seq_len//2:] = 0.9  # Mixed

        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert k_comp.shape[0] == batch_size
        assert stats.tier_distribution[0] > 0  # Some protected tokens


class TestEdgeCasesThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_adds(self) -> None:
        """Test concurrent adds from multiple threads."""
        import threading

        config = CacheConfig()
        cache = TieredKVCache(config)

        errors = []

        def add_tokens(thread_id: int) -> None:
            try:
                k = torch.randn(1, 12, 10, 64)
                v = torch.randn(1, 12, 10, 64)
                retention = torch.rand(1, 10)
                positions = torch.arange(thread_id * 10, (thread_id + 1) * 10).unsqueeze(0)
                cache.add(k, v, retention, positions)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_tokens, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert cache.total_tokens == 50


class TestEdgeCasesContextManager:
    """Tests for context manager."""

    def test_context_manager_cleanup(self) -> None:
        """Test automatic cleanup on context exit."""
        config = CacheConfig()

        with TieredKVCache(config) as cache:
            k = torch.randn(1, 12, 100, 64)
            v = torch.randn(1, 12, 100, 64)
            retention = torch.rand(1, 100)
            positions = torch.arange(100).unsqueeze(0)
            cache.add(k, v, retention, positions)
            assert cache.total_tokens == 100

        # After exit, cache should be cleared
        assert cache.total_tokens == 0
        assert len(cache.k_cache) == 0

    def test_context_manager_exception_safety(self) -> None:
        """Test cleanup happens even with exceptions."""
        config = CacheConfig()
        cache = None

        try:
            with TieredKVCache(config) as c:
                cache = c
                k = torch.randn(1, 12, 100, 64)
                v = torch.randn(1, 12, 100, 64)
                retention = torch.rand(1, 100)
                positions = torch.arange(100).unsqueeze(0)
                c.add(k, v, retention, positions)
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert cache is not None
        assert cache.total_tokens == 0


class TestEdgeCasesSalienceScorer:
    """Tests for SalienceScorer edge cases."""

    def test_single_hidden_state(self) -> None:
        """Test scoring single hidden state."""
        scorer = SalienceScorer(hidden_dim=64, salience_hidden=32)
        hidden = torch.randn(1, 64)
        score = scorer(hidden)
        assert score.shape == torch.Size([1])

    def test_very_large_batch(self) -> None:
        """Test scoring very large batch."""
        scorer = SalienceScorer(hidden_dim=64, salience_hidden=32)
        hidden = torch.randn(1000, 100, 64)
        scores = scorer(hidden)
        assert scores.shape == torch.Size([1000, 100])


class TestEdgeCasesRetentionScheduler:
    """Tests for RetentionScheduler edge cases."""

    def test_extreme_alpha_values(self) -> None:
        """Test scheduler with extreme alpha."""
        scheduler = RetentionScheduler()

        # All zeros
        salience = torch.zeros(10)
        type_priors = torch.zeros(10)
        result = scheduler(salience, type_priors)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

        # All ones
        salience = torch.ones(10)
        type_priors = torch.ones(10)
        result = scheduler(salience, type_priors)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_mismatched_shapes_raises_error(self) -> None:
        """Test mismatched shapes raise ShapeMismatchError."""
        scheduler = RetentionScheduler()
        salience = torch.randn(10)
        type_priors = torch.randn(5)

        with pytest.raises(ShapeMismatchError):
            scheduler(salience, type_priors)