# Author: GC

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    """
    Args:
        vocab_size: size of word vocabulary
        num_tags: total tags
        embed_dim: word embedding dimension
        hidden_dim: output dimension of BiLSTM at each step
        dropout: dropout rate (apply on embeddings)

    Attributes:
        vocab_size: size of word vocabulary
        num_tags: total tags
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        # Layers
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def _get_emissions(
        self, seqs: torch.LongTensor, masks: torch.ByteTensor
    ) -> torch.Tensor:
        """Get emission scores from BiLSTM

        Args:
            seqs: (seq_len, batch_size), sorted by length in descending order
            masks: (seq_len, batch_size), sorted by length in descending order

        Returns:
            emission scores (seq_len, batch_size, num_tags)
        """
        embeds = self.embeds(seqs)  # (seq_len, batch_size, embed_dim)
        embeds = self.dropout(embeds)
        packed = pack_padded_sequence(embeds, masks.sum(0))
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out)  # (seq_len, batch_size, hidden_dim)
        # Space Transform (seq_len, batch_size, num_tags)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(
        self, seqs: torch.LongTensor, tags: torch.LongTensor, masks: torch.ByteTensor
    ) -> torch.Tensor:
        """Negative log likelihood loss
        
        Args:
            seqs: (seq_len, batch_size), sorted by length in descending order
            tags: (seq_len, batch_size), sorted by length in descending order
            masks: (seq_len, batch_size), sorted by length in descending order

        Returns:
            loss
        """
        emissions = self._get_emissions(seqs, masks)
        loss = -self.crf(emissions, tags, mask=masks, reduction="mean")
        return loss

    def decode(
        self, seqs: torch.LongTensor, masks: torch.ByteTensor
    ) -> List[List[int]]:
        """Viterbi decode
        
        Args:
            seqs: (seq_len, batch_size), sorted by length in descending order
            masks: (seq_len, batch_size), sorted by length in descending order

        Returns:
            List of list containing the best tag sequence for each batch
        """
        emissions = self._get_emissions(seqs, masks)
        best_tags = self.crf.decode(emissions, mask=masks)
        return best_tags


if __name__ == "__main__":
    vocab_size = 3
    num_tags = 2
    embed_dim = 10
    hidden_dim = 10
    dropout = 0.1
    model = BiLSTM_CRF(vocab_size, num_tags, embed_dim, hidden_dim, dropout)
    # Inputs are sorted
    seqs = torch.LongTensor([[1, 2, 1], [1, 0, 0]]).t()
    tags = torch.LongTensor([[1, 1, 0], [0, 0, 0]]).t()
    masks = seqs.ne(0)
    print(model.loss(seqs, tags, masks))
    print(model.decode(seqs, masks))
