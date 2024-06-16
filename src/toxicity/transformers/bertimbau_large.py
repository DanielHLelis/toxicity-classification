import polars as pl

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    BertTokenizerFast,
    BertModel,
    PreTrainedTokenizerBase,
    BertForSequenceClassification,
)

PRE_TRAINED_REF = "neuralmind/bert-large-portuguese-cased"
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_LEN = 256


def bert_tokenizer(
    ref: str = PRE_TRAINED_REF,
    truncation: bool = True,
    do_lower_case: bool = True,
    **kwargs
) -> PreTrainedTokenizerBase:
    return BertTokenizerFast.from_pretrained(
        ref, truncation=truncation, do_lower_case=do_lower_case, **kwargs
    )


class BertDataset(Dataset):
    tokenizer: PreTrainedTokenizerBase
    data: pl.DataFrame
    text: pl.Series
    targets: pl.Series
    max_len: int

    def __init__(
        self,
        data_frame: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = DEFAULT_MAX_LEN,
        text_col: str = "text",
        target_col: str = "labels",
    ):
        self.tokenizer = tokenizer
        self.data = data_frame
        self.text = data_frame[text_col]
        self.targets = data_frame[target_col]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "mask": torch.tensor(mask, dtype=torch.int64),
            "targets": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class BertModule(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(BertModule, self).__init__()
        self.bert = BertModel.from_pretrained(
            ref,
            attn_implementation="sdpa",
        )

        hidden_size = self.bert.config.hidden_size

        # Fully connected layer to transform the model features
        self.pre_classifier = torch.nn.Linear(
            hidden_size, hidden_size, dtype=torch.float32
        )

        # Dropout for overfitting prevention
        self.dropout = torch.nn.Dropout(dropout)

        # Final classifier layer to map the transformed feature to the targets
        self.classifier = torch.nn.Linear(
            hidden_size, feature_count, dtype=torch.float32
        )

        # Initialize the classifier weights
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits


class HFBertModule(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(HFBertModule, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            ref,
            num_labels=feature_count,
            hidden_dropout_prob=dropout,
            attn_implementation="sdpa",
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits


class BertDatasetBF16(Dataset):
    tokenizer: PreTrainedTokenizerBase
    data: pl.DataFrame
    text: pl.Series
    targets: pl.Series
    max_len: int

    def __init__(
        self,
        data_frame: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = DEFAULT_MAX_LEN,
        text_col: str = "text",
        target_col: str = "labels",
    ):
        self.tokenizer = tokenizer
        self.data = data_frame
        self.text = data_frame[text_col]
        self.targets = data_frame[target_col]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "mask": torch.tensor(mask, dtype=torch.int64),
            "targets": torch.tensor(self.targets[index], dtype=torch.bfloat16),
        }


class BertModuleBF16(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(BertModuleBF16, self).__init__()
        self.bert = BertModel.from_pretrained(
            ref,
            attn_implementation="sdpa",
        ).bfloat16()

        hidden_size = self.bert.config.hidden_size

        # Fully connected layer to transform the model features
        self.pre_classifier = torch.nn.Linear(
            hidden_size, hidden_size, dtype=torch.bfloat16
        )

        # Dropout for overfitting prevention
        self.dropout = torch.nn.Dropout(dropout)

        # Final classifier layer to map the transformed feature to the targets
        self.classifier = torch.nn.Linear(
            hidden_size, feature_count, dtype=torch.bfloat16
        )

        # Initialize the classifier weights
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits


class HFBertModuleBF16(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(HFBertModuleBF16, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            ref,
            num_labels=feature_count,
            hidden_dropout_prob=dropout,
            attn_implementation="sdpa",
        ).bfloat16()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
