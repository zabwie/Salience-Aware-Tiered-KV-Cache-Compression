"""Training pipeline for SalienceScorer using attention-based supervision."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any
import os
from tqdm import tqdm
from .core import SalienceScorer
from .exceptions import ConfigurationError
class AttentionDataset(Dataset):
    """Dataset for training SalienceScorer on attention patterns.
    Extracts attention weights from language model outputs and pairs them
    with hidden states for supervised training.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        model: Any,
        max_length: int = 512,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prepare_data()
    def _prepare_data(self) -> None:
        """Extract attention patterns from texts."""
        print(f"Extracting attention patterns from {len(self.texts)} texts...")
        for text in tqdm(self.texts, desc="Processing texts"):
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=True
                    )
                hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
                attentions = outputs.attentions[-1]  # [1, num_heads, seq_len, seq_len]
                attention_weights = attentions.mean(dim=1).mean(dim=1).squeeze(0)  # [seq_len]
                mask = inputs.attention_mask.squeeze(0)  # [seq_len]
                attention_weights = attention_weights * mask
                self.samples.append((hidden_states.squeeze(0), attention_weights))
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
        print(f"Prepared {len(self.samples)} training samples")
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
class SalienceTrainer:
    """Trainer for SalienceScorer using attention-based supervision.
    Trains the scorer to predict token importance based on attention patterns
    from a teacher language model.
    """
    def __init__(
        self,
        scorer: SalienceScorer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.scorer = scorer.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            scorer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.
        Returns:
            Average loss for the epoch
        """
        self.scorer.train()
        total_loss = 0.0
        num_batches = 0
        for hidden_states, attention_targets in tqdm(dataloader, desc="Training"):
            hidden_states = hidden_states.to(self.device)
            attention_targets = attention_targets.to(self.device)
            predicted_scores = self.scorer(hidden_states)
            loss = self.criterion(predicted_scores, attention_targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set.
        Returns:
            Average validation loss
        """
        self.scorer.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for hidden_states, attention_targets in tqdm(dataloader, desc="Validating"):
                hidden_states = hidden_states.to(self.device)
                attention_targets = attention_targets.to(self.device)
                predicted_scores = self.scorer(hidden_states)
                loss = self.criterion(predicted_scores, attention_targets)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_dir: str = './checkpoints',
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Train the scorer.
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model based on validation loss
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        history = {
            'train_loss': [],
            'val_loss': []
        }
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_dataloader)
            history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.6f}")
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                history['val_loss'].append(val_loss)
                print(f"Val Loss: {val_loss:.6f}")
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(save_dir, 'salience_scorer_best.pt')
                    self.scorer.save_pretrained(best_path)
                    print(f"Saved best model to {best_path}")
            else:
                last_path = os.path.join(save_dir, 'salience_scorer_last.pt')
                self.scorer.save_pretrained(last_path)
        return history
def train_on_gpt2(
    texts: List[str],
    model_name: str = 'gpt2',
    hidden_dim: int = 768,
    salience_hidden: int = 256,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_path: str = './salience_scorer.pt',
    val_split: float = 0.1
) -> SalienceScorer:
    """Train SalienceScorer on GPT-2 attention patterns.
    This is the main entry point for training a scorer as mentioned in the paper.
    Args:
        texts: List of training texts
        model_name: HuggingFace model name (default: 'gpt2')
        hidden_dim: Hidden dimension (must match model)
        salience_hidden: Hidden dimension for scorer
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        save_path: Path to save trained scorer
        val_split: Fraction of data for validation
    Returns:
        Trained SalienceScorer
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ConfigurationError(
            "transformers library required for training. Install with: pip install transformers",
            "transformers"
        )
    print(f"Loading {model_name} for training...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    val_size = int(len(texts) * val_split)
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    print(f"Training on {len(train_texts)} texts, validating on {len(val_texts)} texts")
    train_dataset = AttentionDataset(train_texts, tokenizer, model)
    val_dataset = AttentionDataset(val_texts, tokenizer, model) if val_texts else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    scorer = SalienceScorer(hidden_dim=hidden_dim, salience_hidden=salience_hidden)
    trainer = SalienceTrainer(scorer, learning_rate=learning_rate)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_dir=os.path.dirname(save_path) or '.',
        save_best=True
    )
    scorer.save_pretrained(save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    return scorer
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 2024, researchers at Stanford University developed a new compression algorithm.",
        "Machine learning models require significant computational resources.",
        "The Eiffel Tower is located in Paris, France.",
        "Attention is all you need.",
    ] * 20  # Repeat for more data
    print("Training SalienceScorer on sample data...")
    trained_scorer = train_on_gpt2(
        texts=sample_texts,
        num_epochs=2,
        batch_size=2,
        save_path='./trained_scorer.pt'
    )
