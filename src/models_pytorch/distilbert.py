import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class DistilBERTClassifier(nn.Module):
    def __init__(self, output_dim: int = 5, dropout: float = 0.3,
                 freeze_bert: bool = False, n_style_features: int = 0):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size + n_style_features, output_dim)

    def forward(self, input_ids, attention_mask, style_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        out = self.dropout(cls_output)

        if style_features is not None:
            out = torch.cat([out, style_features], dim=1)

        out = self.fc(out)
        return out


class DistilBERTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128,
                 style_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.style_features = (
            torch.tensor(style_features, dtype=torch.float32)
            if style_features is not None else None
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.style_features is not None:
            item["style_features"] = self.style_features[idx]
        return item


def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
