from dataclasses import dataclass, field


@dataclass
class BertDataArguments:
    data_name: str = field(default=None)
