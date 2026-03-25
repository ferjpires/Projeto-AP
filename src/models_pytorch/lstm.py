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
        pooling: str = "attention",
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
            dropout=0,
            batch_first=True,
        )
        self.inter_layer_dropout = nn.Dropout(dropout) if n_layers > 1 else None

        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            self.attention = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        mask = (x != 0)

        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)

        if self.pooling == "attention":
            attn_weights = self.attention(output).squeeze(-1)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        elif self.pooling == "mean":
            mask_expanded = mask.unsqueeze(-1).float()
            context = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            hidden_fwd = hidden[-2]
            hidden_bwd = hidden[-1]
            context = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        out = self.dropout(context)
        out = self.fc(out)
        return out
