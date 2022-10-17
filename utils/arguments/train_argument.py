from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class BertTrainingArguments(TrainingArguments):
    cache_dir: str = field(default=None)
    num_proc: int = field(default=None)
