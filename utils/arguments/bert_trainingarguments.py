from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: str = field(default=None)
