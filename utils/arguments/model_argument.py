from dataclasses import dataclass, field


@dataclass
class BertModelArguments:
    num_labels: int = field(default=2)
