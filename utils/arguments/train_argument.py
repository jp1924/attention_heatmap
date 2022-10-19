from transformers import TrainingArguments
from dataclasses import dataclass, field
import os


@dataclass
class BertTrainingArguments(TrainingArguments):
    cache: os.PathLike = field(
        default=None,
        metadata="cache_dir의 경로를 입력합니다.",
    )
    num_proc: int = field(
        default=None,
        metadata="전처리에 사용될 프로세서의 수를 결정합니다.",
    )
    hyperparameter_search: bool = field(
        default=False,
        metadata="huggingface에서 제공하는 hyperparameter_search기능을 사용할지 말지에 대한 결정을 bool값으로 결정합니다.",
    )
