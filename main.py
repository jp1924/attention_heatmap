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

    def metrics(input_values: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, int]:
        """__metrics__ evaluation때 모델의 성능을 검증하기 위한 함수
            trainer의 evaluation_loop시 모델의 성능을 검증하기 위해
            valid데이터의 logits을 전달받아 각종 evaluate의 metric을
            적용시키는 함수 입니다.

        Args:
            input_values (Dict[str, Union[List[int], torch.Tensor]]): evaluation_loop를 거친 logits값을 전달받습니다.

        Returns:
            Dict[str, int]: metrics결과를 끝낸 값을 dict에 넣어 반환합니다.
        """
        predictions = input_values.predictions
        predictions = predictions.argmax(-1)
        references = input_values.label_ids
        result = accuracy._compute(predictions, references, normalize=True)
        return result

    def preprocess(input_data: datasets.Dataset) -> dict:
        """__preprocess__: 각 데이터를 토크나이징 및 별도의 전처리를 적용시키는 함수

        각 데이터의 전처리르 위한 함수 입니다. 이 함수는 datasets으로 부터
        dict형태의 각 데이터를 입력받아 tokenizer로 인코딩 후 dict형식으로 반환합니다.

        Args:
            input_data (datasets.Dataset): Datasets로 부터 건내받은 dict형식의 각 데이터를 전달 받습니다.

        Returns:
            BatchEncoding: Datasets의 각 열의 해당되는 columns를 가지고 전처리 된 데이터를 반환합니다.
        """
        input_text = input_data["document"]
        output_data = tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

    # [NOTE]:
    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name, cache_dir=train_args.cache)
    config = BertConfig(vocab_size=tokenizer.vocab_size, num_labels=model_args.num_labels)
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name, cache_dir=train_args.cache, config=config
    )

    # [NOTE]:
    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=train_args.cache)
    NSMC_train = NSMC_datasets["train"].map(preprocess, num_proc=train_args.num_proc)
    NSMC_valid = NSMC_datasets["test"].map(preprocess, num_proc=train_args.num_proc)

    # [NOTE]:
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
    # [NOTE]:
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
