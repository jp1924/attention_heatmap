from typing import Dict, List, Union
import transformers
import torch
from kobert_tokenizer import KoBertTokenizer


class BertCollator:
    def __init__(self, tokenizer: KoBertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = dict()
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        batch = self.tokenizer.pad(encoded_inputs=input_ids, padding=True, return_attention_mask=True, return_tensors="pt")

        label = [feature["label"] for feature in features]
        batch["labels"] = torch.tensor(label).to(torch.float32)
        batch["output_attentions"] = True

        return batch
