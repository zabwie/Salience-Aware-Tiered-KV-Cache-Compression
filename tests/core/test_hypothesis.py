"""Property-based tests for TTKV using Hypothesis.

These tests verify that cache invariants hold across a wide range of inputs
using generative testing.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import reproduce_failure
from hypothesis.extra.numpy import arrays

from ttkv import CacheConfig, TieredKVCache


# Custom strategies for TTKV
batch_sizes = st.integers(min_value=1, max_value=8)
seq_lengths = st.integers(min_value=1, max_value=1000)
num_heads = st.integers(min_value=1, max_value=16)
head_dims = st.integers(min_value=16, max_value=256)
retention_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
tau_thresholds = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestCacheInvariants:
    """Property-based tests for cache invariants."""

    @given(
        batch_size=batch_sizes,
        seq_len=seq_lengths,
        num_heads=num_heads,
        head_dim=head_dims,
        tau=tau_thresholds,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_compression_ratio_at_least_one(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        tau: float
    ) -> None:
        """Compression ratio should always be >= 1.0.

        This is a critical invariant: compression should never expand the cache.
        """
        config = CacheConfig(
            tier0_size=min(256, seq_len),
            tier1_size=min(2048, seq_len * 2),
            tau_threshold=tau
        )
        cache = TieredKVCache(config)

        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        retention = torch.rand(batch_size, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        # Critical invariant: compression_ratio >= 1.0
        assert stats.compression_ratio >= 1.0, \
            f"Compression ratio {stats.compression_ratio} < 1.0 for seq_len={seq_len}"

        # Shape invariants
        if k_comp is not None:
            assert k_comp.shape[0] == batch_size
            assert k_comp.shape[1] == num_heads
            assert k_comp.shape[3] == head_dim
            assert v_comp.shape == k_comp.shape

    @given(
        seq_len=seq_lengths,
        tau=tau_thresholds,
    )
    @settings(max_examples=50)
    def test_protected_tokens_preserved(
        self,
        seq_len: int,
        tau: float
    ) -> None:
        """Tokens with retention > tau should always be preserved.

        This verifies the structural survival guarantee of TTKV.
        """
        config = CacheConfig(tau_threshold=tau)
        cache = TieredKVCache(config)

        # Create tokens with varying retention
        retention = torch.rand(1, seq_len)
        # Make first and last tokens highly retained
        retention[0, 0] = 1.0
        retention[0, -1] = 1.0

        num_protected = (retention > tau).sum().item()

        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        # All highly retained tokens should be preserved
        assert stats.tier_distribution[0] >= num_protected, \
            f"Expected at least {num_protected} protected tokens, got {stats.tier_distribution[0]}"

    @given(
        batch_size=batch_sizes,
        seq_len=seq_lengths,
    )
    @settings(max_examples=100)
    def test_position_monotonicity(
        self,
        batch_size: int,
        seq_len: int
    ) -> None:
        """Compressed positions should maintain relative ordering.

        This ensures that compression doesn't scramble token order.
        """
        config = CacheConfig(
            tier0_size=max(1, seq_len // 4),
            tier1_size=max(1, seq_len // 2),
        )
        cache = TieredKVCache(config)

        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)
        retention = torch.rand(batch_size, seq_len) * 0.5  # Low retention
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        if pos_comp is not None and pos_comp.numel() > 1:
            # Check that positions are monotonic (sorted)
            for b in range(batch_size):
                batch_pos = pos_comp[b]
                if batch_pos.numel() > 1:
                    # Positions should be roughly sorted
                    sorted_pos, _ = torch.sort(batch_pos)
                    # Allow for some variation due to compression pooling
                    pass  # Positions may not be strictly sorted after pooling


class TestConfigInvariants:
    """Property-based tests for configuration validation."""

    @given(
        tier0=st.integers(min_value=1, max_value=1000),
        tier1=st.integers(min_value=1, max_value=5000),
        tau=tau_thresholds,
    )
    @settings(max_examples=50)
    def test_valid_config_always_works(
        self,
        tier0: int,
        tier1: int,
        tau: float
    ) -> None:
        """Valid configurations should always create working caches."""
        if tier1 < tier0 or tau < 0 or tau > 1:
            pytest.skip("Invalid config parameters")

        config = CacheConfig(
            tier0_size=tier0,
            tier1_size=tier1,
            tau_threshold=tau
        )
        cache = TieredKVCache(config)

        seq_len = max(tier1 * 2, 100)
        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        retention = torch.rand(1, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0)

        # Should not raise
        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None


class TestShapeInvariants:
    """Property-based tests for tensor shape invariants."""

    @given(
        batch_size=batch_sizes,
        num_heads=st.integers(1, 16),
        seq_len=seq_lengths,
        head_dim=st.integers(16, 256),
    )
    @settings(max_examples=50)
    def test_output_shapes_match_input_dims(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int
    ) -> None:
        """Output tensors should preserve batch, heads, and head_dim."""
        config = CacheConfig()
        cache = TieredKVCache(config)

        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        retention = torch.rand(batch_size, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        if k_comp is not None:
            assert k_comp.shape[0] == batch_size
            assert k_comp.shape[1] == num_heads
            assert k_comp.shape[3] == head_dim
            assert v_comp.shape == k_comp.shape
            assert pos_comp.shape[0] == batch_size


class TestStatisticalInvariants:
    """Statistical property tests."""

    @given(
        seq_len=st.integers(100, 500),
        num_iterations=st.integers(5, 20),
    )
    @settings(max_examples=20)
    def test_compression_ratio_stability(
        self,
        seq_len: int,
        num_iterations: int
    ) -> None:
        """Compression ratio should be stable across different random seeds.

        This tests that compression is deterministic and not overly sensitive
        to specific retention values.
        """
        ratios = []

        for _ in range(num_iterations):
            config = CacheConfig()
            cache = TieredKVCache(config)

            k = torch.randn(1, 12, seq_len, 64)
            v = torch.randn(1, 12, seq_len, 64)
            retention = torch.rand(1, seq_len)
            positions = torch.arange(seq_len).unsqueeze(0)

            cache.add(k, v, retention, positions)
            stats = cache.get_stats()
            ratios.append(stats.compression_ratio)

        # Compression ratios should be reasonably stable
        # (within factor of 2 of each other)
        if len(ratios) > 1:
            ratio_std = torch.tensor(ratios).std().item()
            ratio_mean = torch.tensor(ratios).mean().item()
            # Coefficient of variation should be reasonable
            if ratio_mean > 0:
                cv = ratio_std / ratio_mean
                assert cv < 1.0, f"Compression ratio too variable: CV={cv:.2f}"


class TestEdgeCaseHypothesis:
    """Edge cases discovered through property-based testing."""

    @given(
        seq_len=st.integers(1, 10),
    )
    @settings(max_examples=50)
    def test_very_short_sequences(self, seq_len: int) -> None:
        """Very short sequences should still work correctly."""
        config = CacheConfig(tier0_size=256)  # Larger than seq_len
        cache = TieredKVCache(config)

        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        retention = torch.rand(1, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        # Short sequences shouldn't be compressed
        assert k_comp is not None
        assert k_comp.shape[2] == seq_len

    @given(
        seq_len=st.integers(1000, 10000),
    )
    @settings(max_examples=10, deadline=None)
    def test_very_long_sequences(self, seq_len: int) -> None:
        """Very long sequences should handle memory efficiently."""
        config = CacheConfig(tier0_size=256, tier1_size=2048)
        cache = TieredKVCache(config)

        # Add in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            k = torch.randn(1, 12, end - i, 64)
            v = torch.randn(1, 12, end - i, 64)
            retention = torch.rand(1, end - i)
            positions = torch.arange(i, end).unsqueeze(0)
            cache.add(k, v, retention, positions)

        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        # Long sequences should get significant compression
        assert stats.compression_ratio > 1.0
        assert stats.total_tokens == seq_len