from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    
    def __init__(self, ds, tokenizer) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer = tokenizer
        self.cls_token = torch.tensor([tokenizer.token_to_id('[CLS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> Any:
        text = self.ds.loc[index, "text"]
        label = self.ds.loc[index, "sentiment"]
        
        text_enc = self.tokenizer.encode(text).ids
        encoder_input = torch.cat(
            [
                self.cls_token,
                torch.tensor(text_enc, dtype=torch.int64),
            ]
        )
        
        label = torch.tensor(label, dtype=torch.int64)
        
        return {
            "encoder_input": encoder_input,
            # "decoder_input": decoder_input,
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,
            # "src_text": src_text,
            # "tgt_text": tgt_text
        }


def collate_fn(batch):
    
    max_len = min(max([batch[i]["encoder_input"].size(dim=0) for i in range(len(batch))]), 1000)
    # max_len = 200
    
    for i in range(len(batch)):
        batch[i]["encoder_input"] = batch[i]["encoder_input"][:max_len]
        batch[i]["encoder_input"] = torch.cat([batch[i]["encoder_input"], torch.ones(max_len - batch[i]["encoder_input"].size(dim=0), dtype=torch.int64)])
        
    encoder_input = torch.tensor([batch[i]["encoder_input"].tolist() for i in range(len(batch))])
    label = torch.tensor([[batch[i]["label"].item()] for i in range(len(batch))])
    mask = torch.tensor([[1 if ele == 1 else 0 for ele in li] for li in encoder_input])
    mask = (mask > 0).unsqueeze(-1)
    return encoder_input, label.float(), mask