from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing_extensions import Literal


@dataclass
class BertTrainingArguments(TrainingArguments):
    cache: str = field(
        default=None,
        metadata={"help": "cache_dir의 경로를 입력합니다."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "전처리에 사용될 프로세서의 수를 결정합니다."},
    )
    hyperparameter_search: bool = field(
        default=False,
        metadata={"help": "huggingface에서 제공하는 hyperparameter_search기능을 사용할지 말지에 대한 결정을 bool값으로 결정합니다."},
    )
    hp_trial: int = field(
        default=1,
        metadata={"help": "hyperparameter_search시 실행할 시험의 수를 설정합니다."},
    )
    hp_config: str = field(
        default=None,
        metadata={
            "help": """hp실행 시 테스트 할 값들의 범위를 지정합니다. 값의 지정 범위는 각 환경마다 다릅니다.
                       만약 환경에서 제공하는 함수를 사용해야 하는 경우 직접 선언해서 사용해야 합니다."""
        },
    )
    hp_kwargs_config: str = field(
        default=None,
        metadata={"help": "hp를 시도하는 환경마다 넘기는 값들이 전부 다르기에 json파일로 hp환경 설정에 대한 값을 전달합니다."},
    )
    hp_backend: Literal["optuna", "ray", "wandb", "sigopt"] = field(
        default=None,
        metadata={"help": "hp를 시도할 플랫폼을 선택합니다. 플랫폼은 huggingface에서 지원하는 플랫폼만 가능합니다."},
    )
    hp_direction: Literal["minimize", "maximize"] = field(
        default=None,
        metadata={"help": "매트릭 값이 어느 방향으로 수렴할지를 결정합니다."},
    )
