"""Unit tests for input validation in TieredKVCache."""

import pytest
import torch
from ttkv import CacheConfig, TieredKVCache


class TestInputValidation:
    """Tests for input validation in TieredKVCache.add()."""

    @pytest.fixture
    def config(self) -> CacheConfig:
        return CacheConfig()

    @pytest.fixture
    def cache(self, config: CacheConfig) -> TieredKVCache:
        return TieredKVCache(config)

    def test_mismatched_kv_shapes(self, cache: TieredKVCache) -> None:
        """Test error for mismatched key/value shapes."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 32, 64)  # Different seq_len
        retention = torch.rand(1, 64)
        positions = torch.arange(64).unsqueeze(0)

        with pytest.raises(ValueError, match="Keys.*and values.*must have the same shape"):
            cache.add(k, v, retention, positions)

    def test_batch_size_mismatch(self, cache: TieredKVCache) -> None:
        """Test error for batch size mismatch."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(2, 64)  # Different batch size
        positions = torch.arange(64).unsqueeze(0)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            cache.add(k, v, retention, positions)

    def test_sequence_length_mismatch(self, cache: TieredKVCache) -> None:
        """Test error for sequence length mismatch."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(1, 32)  # Different seq length
        positions = torch.arange(64).unsqueeze(0)

        with pytest.raises(ValueError, match="Sequence length mismatch"):
            cache.add(k, v, retention, positions)

    def test_positions_retention_shape_mismatch(self, cache: TieredKVCache) -> None:
        """Test error for positions/retention shape mismatch."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(1, 64)
        positions = torch.arange(32).unsqueeze(0)  # Different shape

        with pytest.raises(ValueError, match="Positions.*and retention.*must have the same shape"):
            cache.add(k, v, retention, positions)

    def test_keys_not_4d(self, cache: TieredKVCache) -> None:
        """Test error for non-4D keys."""
        k = torch.randn(1, 12, 64)  # 3D instead of 4D
        v = torch.randn(1, 12, 64)  # 3D instead of 4D
        retention = torch.rand(1, 64)
        positions = torch.arange(64).unsqueeze(0)

        with pytest.raises(ValueError, match="Keys must be 4D tensor"):
            cache.add(k, v, retention, positions)

    def test_valid_input_passes(self, cache: TieredKVCache) -> None:
        """Test that valid inputs don't raise errors."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(1, 64)
        positions = torch.arange(64).unsqueeze(0)

        # Should not raise
        cache.add(k, v, retention, positions)
        assert cache.total_tokens == 64
