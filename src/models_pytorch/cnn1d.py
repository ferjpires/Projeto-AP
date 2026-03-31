import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        n_filters: int = 100,
        filter_sizes: list[int] | None = None,
        output_dim: int = 5,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor | None = None,
        n_style_features: int = 0,
    ):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes) + n_style_features, output_dim)

    def forward(self, x, style_features=None):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [c.max(dim=2).values for c in conved]

        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)

        if style_features is not None:
            out = torch.cat([out, style_features], dim=1)

        out = self.fc(out)
        return out
