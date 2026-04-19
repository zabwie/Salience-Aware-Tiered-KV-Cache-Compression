"""
Mock type prior implementation without spaCy dependency.
Demonstrates the concept of hardcoded type priors.
"""

import torch
from typing import List, Dict
import re


class MockTypePriorClassifier:
    """
    Mock type prior classifier that assigns retention scores based on token characteristics.
    
    In production, this would use spaCy NER + POS tags.
    Here we use heuristics to approximate the same behavior.
    """
    
    def __init__(self):
        # Common patterns
        self.patterns = {
            'NAMED_ENTITY': [
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
                r'[A-Z]{2,}',                   # ACRONYMS
                r'University|Institute|Research|Google|OpenAI|DeepMind',
            ],
            'NUMERIC': [
                r'\d+',                         # numbers
                r'\d{4}',                       # years
                r'[\d,]+',                     # formatted numbers
            ],
            'PUNCTUATION': [
                r'[.,;:!?()[\]{\}]',
            ],
            'FUNCTION_WORD': [
                r'the|a|an|is|are|was|were|be|been|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can|need|dare|ought|used|to|of|in|for|on|with|at|by|from|as|into|through|during|before|after|above|below|between|under|and|but|or|yet|so|if|because|although|though|while|where|when|that|which|who|whom|whose|what|this|these|those|I|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their|mine|yours|hers|ours|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|someone|somebody|something|anyone|anybody|anything|no one|nobody|nothing|everyone|everybody|everything|all|another|any|both|each|either|few|little|many|more|most|much|neither|none|other|several|some|such|enough|other|same|only|own|very|just|also|even|back|there|here|up|down|out|off|over|under|again|further|then|once|more|most|other|some|such|only|own|same|so|than|too|very|just|now|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very',
            ],
        }
        
        # Retention scores
        self.retention_map = {
            'NAMED_ENTITY': 1.0,
            'NUMERIC': 1.0,
            'PUNCTUATION': 0.05,
            'FUNCTION_WORD': 0.05,
            'CONTENT_WORD': 0.3,  # default
        }
    
    def classify_tokens(self, tokens: List[str]) -> Dict[str, float]:
        """
        Classify tokens and assign retention scores.
        
        Args:
            tokens: List of token strings
        
        Returns:
            Dict mapping token index to retention score
        """
        retention_scores = {}
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Check each pattern
            matched = False
            
            # Named entities (heuristic: capitalized words)
            if re.match(r'^[A-Z]', token) and len(token) > 1:
                retention_scores[i] = self.retention_map['NAMED_ENTITY']
                matched = True
            
            # Numbers
            elif re.match(r'^\d', token):
                retention_scores[i] = self.retention_map['NUMERIC']
                matched = True
            
            # Punctuation
            elif re.match(r'^[.,;:!?()[\]{\}]$', token):
                retention_scores[i] = self.retention_map['PUNCTUATION']
                matched = True
            
            # Function words (stop words)
            elif token_lower in self.patterns['FUNCTION_WORD'][0].split('|'):
                retention_scores[i] = self.retention_map['FUNCTION_WORD']
                matched = True
            
            # Default: content word
            if not matched:
                retention_scores[i] = self.retention_map['CONTENT_WORD']
        
        return retention_scores
    
    def get_retention_tensor(self, token_ids: torch.Tensor, vocab_mapping: Dict[int, str] = None) -> torch.Tensor:
        """
        Get retention tensor for token IDs.
        
        Args:
            token_ids: [batch, seq_len] tensor of token IDs
            vocab_mapping: Optional mapping from token IDs to strings
        
        Returns:
            [batch, seq_len] retention scores
        """
        batch_size, seq_len = token_ids.shape
        retention = torch.full((batch_size, seq_len), 0.3)  # default
        
        if vocab_mapping is None:
            # Without vocab mapping, use position-based heuristics
            # First and last 10% get higher retention
            for b in range(batch_size):
                important_positions = set(range(0, seq_len // 10)) | set(range(seq_len * 9 // 10, seq_len))
                for i in range(seq_len):
                    if i in important_positions:
                        retention[b, i] = 0.9
                    elif i % 10 == 0:  # Every 10th token
                        retention[b, i] = 0.7
        else:
            # With vocab mapping, do proper classification
            for b in range(batch_size):
                tokens = [vocab_mapping.get(int(token_ids[b, i]), f"token_{token_ids[b, i]}") 
                         for i in range(seq_len)]
                scores = self.classify_tokens(tokens)
                for i, score in scores.items():
                    retention[b, i] = score
        
        return retention


def create_mock_retention(seq_len: int, num_named_entities: int = 20, 
                          num_numbers: int = 10) -> torch.Tensor:
    """
    Create mock retention scores that simulate spaCy NER/POS output.
    
    Args:
        seq_len: Sequence length
        num_named_entities: Number of named entity positions
        num_numbers: Number of numeric positions
    
    Returns:
        [1, seq_len] retention tensor
    """
    retention = torch.full((1, seq_len), 0.3)  # content words
    
    # Randomly select positions for named entities
    ne_positions = torch.randperm(seq_len)[:num_named_entities]
    retention[0, ne_positions] = 1.0
    
    # Randomly select positions for numbers
    num_positions = torch.randperm(seq_len)[:num_numbers]
    retention[0, num_positions] = 1.0
    
    # First and last tokens get higher retention
    retention[0, :seq_len//20] = 0.9
    retention[0, -seq_len//20:] = 0.9
    
    # Every 10th token gets moderate retention (simulates content words)
    for i in range(0, seq_len, 10):
        if retention[0, i] < 0.5:
            retention[0, i] = 0.7
    
    return retention


def compute_type_prior_retention(token_ids: torch.Tensor, 
                               alpha: float = 0.0) -> torch.Tensor:
    """
    Compute retention scores using hardcoded type priors.
    
    This replaces the learned salience scorer.
    Set alpha = 0.0 for pure type prior (no learned component).
    
    Args:
        token_ids: [batch, seq_len]
        alpha: blending parameter (0.0 = pure type prior)
    
    Returns:
        [batch, seq_len] retention scores
    """
    batch_size, seq_len = token_ids.shape
    
    # Create mock retention (simulating spaCy NER/POS output)
    retention = create_mock_retention(seq_len)
    retention = retention.repeat(batch_size, 1)
    
    return retention


# Export for use in benchmark
if __name__ == "__main__":
    # Test
    seq_len = 100
    token_ids = torch.randint(0, 50000, (1, seq_len))
    retention = compute_type_prior_retention(token_ids)
    
    print(f"Created retention scores for {seq_len} tokens")
    print(f"Stats: min={retention.min():.2f}, max={retention.max():.2f}, mean={retention.mean():.2f}")
    print(f"High retention (>0.8): {(retention > 0.8).sum().item()} tokens")
    print(f"Low retention (<0.2): {(retention < 0.2).sum().item()} tokens")
