"""Unit tests for TTKV core functionality.

These are fast-running unit tests that verify individual components
without requiring model downloads or heavy computation.
"""

import pytest
import torch
from ttkv import (
    CacheConfig,
    TieredKVCache,
    SalienceScorer,
    RetentionScheduler,
)


class TestCacheConfig:
    """Unit tests for CacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()

        assert config.hidden_dim == 768
        assert config.num_heads == 12
        assert config.head_dim == 64
        assert config.tier0_size == 256
        assert config.tier1_size == 2048
        assert config.tier1_compression == 4
        assert config.tier2_compression == 16
        assert config.tau_threshold == 0.8

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            hidden_dim=1024,
            num_heads=16,
            tier0_size=512,
            tier1_size=4096,
            tau_threshold=0.9,
        )

        assert config.hidden_dim == 1024
        assert config.num_heads == 16
        assert config.tier0_size == 512
        assert config.tier1_size == 4096
        assert config.tau_threshold == 0.9

    def test_type_priors_default(self) -> None:
        """Test default type priors are set correctly."""
        config = CacheConfig()

        assert config.type_priors['NAMED_ENTITY'] == 1.0
        assert config.type_priors['NUMERIC'] == 1.0
        assert config.type_priors['CONTENT_WORD'] == 0.7
        assert config.type_priors['FUNCTION_WORD'] == 0.1
        assert config.type_priors['PUNCTUATION'] == 0.0

    def test_type_priors_custom(self) -> None:
        """Test custom type priors."""
        config = CacheConfig(
            type_priors={'CUSTOM': 0.5, 'OTHER': 0.3}
        )

        assert config.type_priors['CUSTOM'] == 0.5
        assert config.type_priors['OTHER'] == 0.3


class TestSalienceScorer:
    """Unit tests for SalienceScorer neural network."""

    @pytest.fixture
    def scorer(self) -> SalienceScorer:
        return SalienceScorer(hidden_dim=768, salience_hidden=256)

    def test_initialization(self, scorer: SalienceScorer) -> None:
        """Test scorer initializes correctly."""
        assert isinstance(scorer.net, torch.nn.Sequential)

    def test_forward_single(self, scorer: SalienceScorer) -> None:
        """Test forward pass with single token."""
        hidden = torch.randn(1, 768)
        score = scorer(hidden)

        assert score.shape == torch.Size([1])
        assert torch.isfinite(score).all()

    def test_forward_batch(self, scorer: SalienceScorer) -> None:
        """Test forward pass with batch."""
        batch_size = 4
        seq_len = 10
        hidden = torch.randn(batch_size, seq_len, 768)
        scores = scorer(hidden)

        assert scores.shape == torch.Size([batch_size, seq_len])
        assert torch.isfinite(scores).all()

    def test_forward_deterministic_in_eval(self, scorer: SalienceScorer) -> None:
        """Test that same input produces same output in eval mode (no dropout)."""
        scorer.eval()
        hidden = torch.randn(2, 5, 768)
        score1 = scorer(hidden)
        score2 = scorer(hidden)

        assert torch.allclose(score1, score2)

    def test_dropout_in_eval(self, scorer: SalienceScorer) -> None:
        """Test dropout is disabled in eval mode."""
        scorer.eval()
        hidden = torch.randn(2, 5, 768)
        score1 = scorer(hidden)
        score2 = scorer(hidden)

        assert torch.allclose(score1, score2)


class TestRetentionScheduler:
    """Unit tests for RetentionScheduler."""

    @pytest.fixture
    def scheduler(self) -> RetentionScheduler:
        return RetentionScheduler()

    def test_initialization(self, scheduler: RetentionScheduler) -> None:
        """Test scheduler initializes with learnable alpha."""
        assert isinstance(scheduler.alpha, torch.nn.Parameter)

    def test_forward_combination(self, scheduler: RetentionScheduler) -> None:
        """Test that salience and type priors are combined."""
        salience = torch.tensor([0.5, 0.8, 0.3])
        type_priors = torch.tensor([0.2, 0.9, 0.4])

        result = scheduler(salience, type_priors)

        assert result.shape == salience.shape
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_alpha_bounds(self, scheduler: RetentionScheduler) -> None:
        """Test that alpha is constrained to (0, 1) via sigmoid."""
        salience = torch.tensor([0.5, 0.5])
        type_priors = torch.tensor([0.5, 0.5])

        result = scheduler(salience, type_priors)
        assert torch.all(result >= 0) and torch.all(result <= 1)


class TestTieredKVCacheBasic:
    """Basic unit tests for TieredKVCache."""

    @pytest.fixture
    def config(self) -> CacheConfig:
        return CacheConfig(
            tier0_size=128,
            tier1_size=512,
            tier1_compression=4,
            tier2_compression=16,
        )

    @pytest.fixture
    def cache(self, config: CacheConfig) -> TieredKVCache:
        return TieredKVCache(config)

    def test_initialization(self, cache: TieredKVCache, config: CacheConfig) -> None:
        """Test cache initializes correctly."""
        assert cache.config == config
        assert cache.total_tokens == 0
        assert len(cache.k_cache) == 0

    def test_clear(self, cache: TieredKVCache) -> None:
        """Test clear resets all state."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(1, 64)
        positions = torch.arange(64).unsqueeze(0)

        cache.add(k, v, retention, positions)
        cache.clear()

        assert cache.total_tokens == 0
        assert len(cache.k_cache) == 0

    def test_empty_cache_returns_none(self, cache: TieredKVCache) -> None:
        """Test that empty cache returns None."""
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        assert k_comp is None
        assert v_comp is None
        assert pos_comp is None

    def test_single_add(self, cache: TieredKVCache) -> None:
        """Test adding single chunk."""
        seq_len = 64
        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        retention = torch.rand(1, seq_len)
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)

        assert cache.total_tokens == seq_len

    def test_multiple_adds(self, cache: TieredKVCache) -> None:
        """Test adding multiple chunks."""
        chunk_size = 64
        for i in range(3):
            k = torch.randn(1, 12, chunk_size, 64)
            v = torch.randn(1, 12, chunk_size, 64)
            retention = torch.rand(1, chunk_size)
            positions = torch.arange(i * chunk_size, (i + 1) * chunk_size).unsqueeze(0)
            cache.add(k, v, retention, positions)

        assert cache.total_tokens == 3 * chunk_size

    def test_stats_empty_cache(self, cache: TieredKVCache) -> None:
        """Test stats on empty cache."""
        stats = cache.get_stats()
        assert stats['total_tokens'] == 0
        assert stats['compression_ratio'] == 1.0

    def test_small_sequence_no_compression(self, cache: TieredKVCache) -> None:
        """Test small sequences don't get compressed."""
        seq_len = 64
        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        retention = torch.rand(1, seq_len) * 0.5
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert stats['total_tokens'] == seq_len
        assert stats['compression_ratio'] == 1.0

    def test_large_sequence_compression(self, cache: TieredKVCache) -> None:
        """Test large sequences get compressed."""
        seq_len = 1024
        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        retention = torch.rand(1, seq_len) * 0.5
        positions = torch.arange(seq_len).unsqueeze(0)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert stats['total_tokens'] == seq_len
        assert stats['compression_ratio'] > 1.0
        assert stats['compressed_tokens'] < seq_len

    def test_protected_tokens_preserved(self, cache: TieredKVCache) -> None:
        """Test tokens with retention > tau are preserved."""
        seq_len = 512
        k = torch.randn(1, 12, seq_len, 64)
        v = torch.randn(1, 12, seq_len, 64)
        positions = torch.arange(seq_len).unsqueeze(0)

        # Make some tokens highly retained
        retention = torch.ones(1, seq_len) * 0.5
        retention[0, 0] = 0.95  # Protected
        retention[0, 100] = 0.95  # Protected

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
        assert pos_comp is not None

    def test_error_handling_mismatched_kv_shapes(self, cache: TieredKVCache) -> None:
        """Test error handling for mismatched k/v shapes."""
        k = torch.randn(1, 12, 64, 64)
        v = torch.randn(1, 12, 64, 64)
        retention = torch.rand(1, 64)
        positions = torch.arange(64).unsqueeze(0)

        cache.add(k, v, retention, positions)

        k2 = torch.randn(1, 8, 64, 64)
        v2 = torch.randn(1, 8, 64, 64)
        retention2 = torch.rand(1, 64)
        positions2 = torch.arange(64).unsqueeze(0)

        cache.add(k2, v2, retention2, positions2)
        with pytest.raises(RuntimeError):
            cache.get_compressed_cache()


class TestTieredKVCacheMultiBatch:
    """Tests for multi-batch scenarios."""

    @pytest.fixture
    def config(self) -> CacheConfig:
        return CacheConfig(
            tier0_size=128,
            tier1_size=512,
        )

    @pytest.fixture
    def cache(self, config: CacheConfig) -> TieredKVCache:
        return TieredKVCache(config)

    def test_two_batch_compression(self, cache: TieredKVCache) -> None:
        """Test compression with batch size 2."""
        batch_size = 2
        seq_len = 256

        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)
        retention = torch.rand(batch_size, seq_len) * 0.5
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()

        assert k_comp is not None
        assert k_comp.shape[0] == batch_size
        assert stats['total_tokens'] == seq_len

    def test_batch_with_different_retention(self, cache: TieredKVCache) -> None:
        """Test batch with different retention per sample."""
        batch_size = 2
        seq_len = 256

        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)

        retention = torch.zeros(batch_size, seq_len)
        retention[0, :] = 0.3
        retention[1, :] = 0.9

        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cache.add(k, v, retention, positions)
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()

        assert k_comp is not None
