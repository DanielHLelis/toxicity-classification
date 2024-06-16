import polars as pl

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    XLMRobertaTokenizerFast, XLMRobertaModel, PreTrainedTokenizerBase,
    XLMRobertaForSequenceClassification
    )

PRE_TRAINED_REF = 'FacebookAI/xlm-roberta-base'
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_LEN = 256


def xlm_roberta_tokenizer(
    ref: str = PRE_TRAINED_REF,
    truncation: bool = True,
    do_lower_case: bool = True,
    **kwargs
) -> PreTrainedTokenizerBase:
    return XLMRobertaTokenizerFast.from_pretrained(
        ref, truncation=truncation, do_lower_case=do_lower_case, **kwargs
    )


class XLMRobertaDataset(Dataset):
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


class XLMRobertaModule(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(XLMRobertaModule, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            ref,
        )
        
        hidden_size = self.xlm_roberta.config.hidden_size

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
        output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits


class HFXLMRobertaModule(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(HFXLMRobertaModule, self).__init__()
        self.xlm_roberta = XLMRobertaForSequenceClassification.from_pretrained(
            ref,
            num_labels=feature_count,
            hidden_dropout_prob=dropout,
        )

    def forward(self, input_ids, attention_mask):
        output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

class XLMRobertaDatasetBF16(Dataset):
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


class XLMRobertaModuleBF16(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(XLMRobertaModuleBF16, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            ref,
        ).bfloat16()

        hidden_size = self.xlm_roberta.config.hidden_size

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
        output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits


class HFXLMRobertaModuleBF16(nn.Module):
    def __init__(
        self,
        feature_count: int,
        dropout: float = DEFAULT_DROPOUT,
        ref: str = PRE_TRAINED_REF,
    ):
        super(HFXLMRobertaModuleBF16, self).__init__()
        self.xlm_roberta = XLMRobertaForSequenceClassification.from_pretrained(
            ref,
            num_labels=feature_count,
            hidden_dropout_prob=dropout,
        ).bfloat16()

    def forward(self, input_ids, attention_mask):
        output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
