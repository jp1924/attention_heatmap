from transformers import TrainingArguments
from dataclasses import dataclass, field
import os

@dataclass
class StudyForTrainingArguments(TrainingArguments):
    cache_dir: str = field(
        default=None
    )
