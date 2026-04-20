"""Named Entity Recognition classifier using HuggingFace transformers."""

import torch
from typing import List, Dict, Optional, Any, Tuple
import re

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class NERClassifier:
    """Real NER classifier using HuggingFace transformers.

    Uses a pre-trained token classification model to identify named entities,
    numeric values, and other important token types for retention scoring.

    Falls back to rule-based classification if transformers not available.
    """

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.device = device
        self.model_name = model_name
        self._pipeline: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self._pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if device == 'cuda' else -1,
                    aggregation_strategy="simple"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Loaded NER model: {model_name}")
            except Exception as e:
                print(f"Failed to load NER model: {e}. Using fallback classification.")
                self._pipeline = None

        # Retention score mapping
        self.retention_map = {
            'NAMED_ENTITY': 1.0,
            'NUMERIC': 1.0,
            'PUNCTUATION': 0.05,
            'FUNCTION_WORD': 0.05,
            'CONTENT_WORD': 0.3,
        }

    def classify_tokens(self, text: str) -> List[Tuple[str, str, float]]:
        """Classify tokens in text.

        Args:
            text: Input text to classify

        Returns:
            List of (token, label, retention_score) tuples
        """
        if self._pipeline is not None:
            return self._classify_with_model(text)
        else:
            return self._classify_with_rules(text)

    def _classify_with_model(self, text: str) -> List[Tuple[str, str, float]]:
        """Classify using HuggingFace NER model."""
        try:
            results = self._pipeline(text)

            # Map NER labels to our retention categories
            label_mapping = {
                'PER': 'NAMED_ENTITY',
                'ORG': 'NAMED_ENTITY',
                'LOC': 'NAMED_ENTITY',
                'MISC': 'NAMED_ENTITY',
            }

            classifications = []
            for result in results:
                word = result['word']
                entity = result.get('entity_group', result.get('entity', 'O'))

                # Map to our categories
                if entity in label_mapping:
                    category = label_mapping[entity]
                elif entity == 'O':
                    # Check if numeric
                    if re.match(r'^\d', word):
                        category = 'NUMERIC'
                    elif word.lower() in self._get_function_words():
                        category = 'FUNCTION_WORD'
                    else:
                        category = 'CONTENT_WORD'
                else:
                    category = 'NAMED_ENTITY'

                score = self.retention_map.get(category, 0.3)
                classifications.append((word, category, score))

            return classifications

        except Exception as e:
            print(f"NER classification failed: {e}. Falling back to rules.")
            return self._classify_with_rules(text)

    def _classify_with_rules(self, text: str) -> List[Tuple[str, str, float]]:
        """Fallback rule-based classification."""
        tokens = text.split()
        classifications = []

        function_words = self._get_function_words()

        for token in tokens:
            token_clean = re.sub(r'[^\w\s]', '', token)
            token_lower = token_clean.lower()

            if not token_clean:
                continue

            # Named entity: capitalized multi-word
            if re.match(r'^[A-Z][a-z]+$', token) and len(token) > 1:
                category = 'NAMED_ENTITY'
            # Numeric
            elif re.match(r'^\d', token):
                category = 'NUMERIC'
            # Punctuation
            elif re.match(r'^[.,;:!?()[\]{}]$', token):
                category = 'PUNCTUATION'
            # Function word
            elif token_lower in function_words:
                category = 'FUNCTION_WORD'
            else:
                category = 'CONTENT_WORD'

            score = self.retention_map.get(category, 0.3)
            classifications.append((token, category, score))

        return classifications

    def _get_function_words(self) -> set:
        """Get set of function words."""
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and',
            'but', 'or', 'yet', 'so', 'if', 'that', 'which', 'who', 'whom',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        }
        return function_words

    def get_retention_scores(self, text: str) -> torch.Tensor:
        """Get retention scores for text as tensor.

        Args:
            text: Input text

        Returns:
            Tensor of retention scores [seq_len]
        """
        classifications = self.classify_tokens(text)
        scores = [score for _, _, score in classifications]
        return torch.tensor(scores, dtype=torch.float32)


class TokenClassifier:
    """Unified token classifier combining NER and POS tagging."""

    def __init__(
        self,
        ner_model: str = "dslim/bert-base-NER",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.device = device
        self.ner = NERClassifier(ner_model, device)

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text and return structured results.

        Args:
            text: Input text

        Returns:
            Dictionary with tokens, labels, and retention scores
        """
        classifications = self.ner.classify_tokens(text)

        tokens = [token for token, _, _ in classifications]
        labels = [label for _, label, _ in classifications]
        scores = [score for _, _, score in classifications]

        return {
            'tokens': tokens,
            'labels': labels,
            'retention_scores': torch.tensor(scores, dtype=torch.float32),
            'named_entities': [t for t, l, _ in classifications if l == 'NAMED_ENTITY'],
            'numeric_values': [t for t, l, _ in classifications if l == 'NUMERIC'],
        }


if __name__ == "__main__":
    # Example usage
    text = "In 2024, researchers at Stanford University developed a new compression algorithm."

    print("Testing NER Classifier:")
    print(f"Text: {text}\n")

    classifier = TokenClassifier()
    result = classifier.classify(text)

    print(f"Tokens: {result['tokens']}")
    print(f"Labels: {result['labels']}")
    print(f"Retention Scores: {result['retention_scores']}")
    print(f"Named Entities: {result['named_entities']}")
    print(f"Numeric Values: {result['numeric_values']}")
