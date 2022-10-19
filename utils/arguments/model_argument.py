from dataclasses import dataclass, field


@dataclass
class BertModelArguments:
    num_labels: int = field(
        default=2,
        metadata="classification에 사용될 labels의 개수를 설정합니다. 만약 multi-classification인데 num_labels가 binary라면 학습이 안될 수 있습니다.",
    )
