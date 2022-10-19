from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Literal
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
    hp_trial: int = field(
        default=None,
        metadata="hyperparameter_search시 실행할 시험의 수를 설정합니다.",
    )
    hp_config: os.PathLike = field(
        default=None,
        metadata="hyperparameter_searcht설정 값을 담고 있는 json파일의 경로를 할당할 수 있습니다. json",
    )
    hp_backend: Literal["optuna", "ray", "wandb", "sigopt"] = field(
        default=None,
        metadata="hp를 시도할 플랫폼을 선택합니다. 플랫폼은 huggingface에서 지원하는 플랫폼만 가능합니다.",
    )
    hp_direction: Literal["minimize", "maximize"] = field(
        default=None,
        metadata="매트릭 값이 어느 방향으로 수렴할지를 결정합니다.",
    )
