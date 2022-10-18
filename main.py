import os
from argparse import Namespace
from typing import Dict, List, Union

import datasets
import torch
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    BatchEncoding,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    HfArgumentParser,
    Trainer,
)
from transformers.integrations import WandbCallback

from data import BertHeatmapCollator
from trainer import HeatmapTrainer
from utils import BertModelArguments, BertTrainingArguments


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)  # [TODO]: trainginarguemnt의 run_name으로 변경

    def metrics(outputs: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, int]:
        """_summary_

        Args:
            outputs (Dict[str, Union[List[int], torch.Tensor]]): _description_

        Returns:
            Dict[str, int]: _description_
        """
        predictions = outputs.predictions
        predictions = predictions.argmax(-1)
        references = outputs.label_ids
        result = accuracy._compute(predictions, references, normalize=True)
        return result

    def preprocess(input_data: datasets.Dataset) -> BatchEncoding:
        """_summary_

        Args:
            input_data (datasets.Dataset): _description_

        Returns:
            BatchEncoding: _description_
        """
        input_text = input_data["document"]
        output_data = tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name, cache_dir=train_args.cache)
    config = BertConfig(vocab_size=tokenizer.vocab_size, num_labels=model_args.num_labels)
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name, cache_dir=train_args.cache, config=config
    )

    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=train_args.cache)
    NSMC_train = NSMC_datasets["train"].map(preprocess, num_proc=train_args.num_proc)
    NSMC_valid = NSMC_datasets["test"].map(preprocess, num_proc=train_args.num_proc)

    accuracy = load("accuracy")
    callback_func = [WandbCallback] if os.getenv("WANDB_DISABLED") != "true" else None
    collator = BertHeatmapCollator(tokenizer)

    trainer = HeatmapTrainer(
        model=model,
        args=train_args,
        compute_metrics=metrics,
        train_dataset=NSMC_train,
        eval_dataset=NSMC_valid,
        data_collator=collator,
        callbacks=callback_func,
        tokenizer=tokenizer,
    )
    # do train, do eval, do predict으로 변경
    trainer.train(ignore_keys_for_eval=["attentions"])
    if train_args.do_train:
        train(trainer, train_args)
    if train_args.do_eval:
        eval(trainer, NSMC_valid)
    if train_args.do_predict:
        predict(trainer, NSMC_valid)


def train(trainer: Trainer, args: Namespace) -> None:
    """_summary_

    Args:
        trainer (Trainer): _description_
        args (Namespace): _description_
    """
    trainer


def eval(trainer: Trainer, valid_data: datasets.Dataset) -> None:
    """_summary_

    Args:
        trainer (Trainer): _description_
        valid_data (datasets.Dataset): _description_
    """
    trainer


def predict(trainer: Trainer, test_data: datasets.Dataset) -> None:
    """_summary_

    Args:
        trainer (Trainer): _description_
        test_data (datasets.Dataset): _description_
    """
    trainer


if "__main__" == __name__:
    # [TODO]: 경로 관련 설정 삭제, 환경 변수 launch.json, shell script로 변경
    parser = HfArgumentParser(BertTrainingArguments, BertModelArguments)

    main(parser)
