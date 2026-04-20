"""Unit tests for SalienceScorer pre-trained weights loading."""

import os
import tempfile
import pytest
import torch
from ttkv import SalienceScorer


class TestPretrainedScorer:
    """Tests for pre-trained scorer weights functionality."""

    @pytest.fixture
    def scorer(self) -> SalienceScorer:
        return SalienceScorer(hidden_dim=768, salience_hidden=256)

    @pytest.fixture
    def temp_weights_file(self) -> str:
        """Create a temporary weights file."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        yield path
        os.unlink(path)

    def test_save_pretrained(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test saving pre-trained weights."""
        scorer.save_pretrained(temp_weights_file)
        assert os.path.exists(temp_weights_file)

        checkpoint = torch.load(temp_weights_file)
        assert 'hidden_dim' in checkpoint
        assert 'salience_hidden' in checkpoint
        assert 'state_dict' in checkpoint
        assert checkpoint['hidden_dim'] == 768
        assert checkpoint['salience_hidden'] == 256

    def test_load_pretrained(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test loading pre-trained weights."""
        # Set a specific weight value
        with torch.no_grad():
            scorer.net[0].weight.fill_(0.5)

        scorer.save_pretrained(temp_weights_file)

        # Create new scorer and load weights
        new_scorer = SalienceScorer(hidden_dim=768, salience_hidden=256)
        new_scorer.load_pretrained(temp_weights_file)

        # Verify weights were loaded
        assert torch.allclose(new_scorer.net[0].weight, torch.tensor(0.5))

    def test_load_pretrained_wrong_hidden_dim(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test loading weights with mismatched hidden dimension."""
        scorer.save_pretrained(temp_weights_file)

        wrong_scorer = SalienceScorer(hidden_dim=512, salience_hidden=256)
        with pytest.raises(RuntimeError, match="Hidden dimension mismatch"):
            wrong_scorer.load_pretrained(temp_weights_file)

    def test_load_pretrained_wrong_salience_hidden(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test loading weights with mismatched salience hidden dimension."""
        scorer.save_pretrained(temp_weights_file)

        wrong_scorer = SalienceScorer(hidden_dim=768, salience_hidden=128)
        with pytest.raises(RuntimeError, match="Salience hidden dimension mismatch"):
            wrong_scorer.load_pretrained(temp_weights_file)

    def test_from_pretrained(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test from_pretrained class method."""
        # Set a specific weight value
        with torch.no_grad():
            scorer.net[0].weight.fill_(0.75)

        scorer.save_pretrained(temp_weights_file)

        # Load using class method
        loaded_scorer = SalienceScorer.from_pretrained(temp_weights_file)

        assert loaded_scorer.hidden_dim == 768
        assert loaded_scorer.salience_hidden == 256
        assert torch.allclose(loaded_scorer.net[0].weight, torch.tensor(0.75))

    def test_from_pretrained_not_found(self) -> None:
        """Test error when weights file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Pre-trained weights not found"):
            SalienceScorer.from_pretrained("/nonexistent/path.pt")

    def test_load_pretrained_not_found(self, scorer: SalienceScorer) -> None:
        """Test error when loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Pre-trained weights not found"):
            scorer.load_pretrained("/nonexistent/path.pt")

    def test_weights_preserved_after_load(self, scorer: SalienceScorer, temp_weights_file: str) -> None:
        """Test that forward pass works correctly after loading weights."""
        # Set deterministic weights
        torch.manual_seed(42)
        with torch.no_grad():
            for param in scorer.parameters():
                param.normal_(0, 0.1)

        scorer.save_pretrained(temp_weights_file)

        # Load and test forward pass
        loaded_scorer = SalienceScorer.from_pretrained(temp_weights_file)
        loaded_scorer.eval()

        hidden = torch.randn(2, 10, 768)
        scores = loaded_scorer(hidden)

        assert scores.shape == torch.Size([2, 10])
        assert torch.isfinite(scores).all()
