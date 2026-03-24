import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 5,
        n_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor | None = None,
        pooling: str = "attention",  # "attention", "mean", or "last"
    ):
        super().__init__()
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            self.attention = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != 0)  # padding mask

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch, seq_len, hidden*2)

        if self.pooling == "attention":
            # Attention pooling
            attn_weights = self.attention(output).squeeze(-1)  # (batch, seq_len)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)
            context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        elif self.pooling == "mean":
            # Mean pooling (excluding padding)
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            context = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Last hidden state (original behavior)
            hidden_fwd = hidden[-2]
            hidden_bwd = hidden[-1]
            context = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        out = self.dropout(context)
        out = self.fc(out)
        return out
