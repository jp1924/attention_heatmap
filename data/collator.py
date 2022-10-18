from typing import Dict, List, Union
from transformers import PreTrainedTokenizer
import torch


class BertHeatmapCollator:
    tokenizer: PreTrainedTokenizer
    outputs_attentions: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """들어온 데이터의 길이를 일정하게 만드는 class 입니다.

            Collator는 Trainer 혹은 Dataloader로 부터 전달받은 데이터들을
            하나로 모은 뒤 길이를 일정하게 만들어 주는 기능을 수앻합니다.
            길이를 일정하게 만들기 위해 tokenizer의 <pad>토큰을 사용해 길이를 일정하게 만듭니다.

            혹은 evaluation_loop와의 연계를 위해 pad를 -100으로도 할수 있습니다.
        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): Trainer의 Dataloader로 부터 건내받은 데이터를 전달받습니다.

        Returns:
            Dict[str, torch.Tensor]: pad처리한 데이터를 dict형태로 반환합니다. 이 때 각 dict의 key 값은
                                     model.forward의 매개변수와 이름이 일치해야 합니다.
                                     그렇지 않으면 값이 자동으로 삭제됩니다.
        """

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        batch = self.tokenizer.pad(
            encoded_inputs=input_ids, padding=True, return_attention_mask=True, return_tensors="pt"
        )

        labels = [feature["label"] for feature in features]
        batch["labels"] = torch.tensor(labels).to(torch.long)
        batch["output_attentions"] = self.outputs_attentions

        return batch
