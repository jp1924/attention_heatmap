from dataclasses import dataclass, field


@dataclass
class BertModelArguments:
    num_labels: int = field(
        default=2,
        metadata={
            "help": "classification에 사용될 labels의 개수를 설정합니다. 만약 multi-classification인데 num_labels가 binary라면 학습이 안될 수 있습니다."
        },
    )
    model_name: str = field(
        default=None,
        metadata={"help": "huggingface허브, 로컬로 부터 불러올 데이터의 경로를 입력합니다."},
    )
