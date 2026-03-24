import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class DistilBERTClassifier(nn.Module):
    def __init__(self, output_dim: int = 5, dropout: float = 0.3, freeze_bert: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)  # 768 → 5

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        out = self.dropout(cls_output)
        out = self.fc(out)
        return out


class DistilBERTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

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
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
