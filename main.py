import os
from typing import Dict, List, Union

import datasets
import torch
from setproctitle import setproctitle
from transformers import BatchEncoding, BertConfig, BertForSequenceClassification, HfArgumentParser
from transformers.integrations import WandbCallback

from argument import CustomTrainingArguments
from collator import BertCollator
from trainer import CustomTrainer


def main(parser: HfArgumentParser) -> None:
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle("[JP]test")  # [TODO]: trainginarguemnt의 run_name으로 변경

    def metrics(outputs: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, int]:
        predictions = outputs.predictions
        predictions = predictions.argmax(-1)
        references = outputs.label_ids
        result = accuracy._compute(predictions, references, normalize=True)
        return result

    def preprocess(input_data: datasets.Dataset) -> BatchEncoding:
        input_text = input_data["document"]
        output_data = bert_tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

    # [TODO]: 모델 monologg/kobert에서 klue-bert로 변경
    bert_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", cache_dir=train_args.cache_dir)
    config = BertConfig(vocab_size=bert_tokenizer.vocab_size, num_labels=2)  # [TODO]: num_labels argument로 빼기
    model = BertForSequenceClassification.from_pretrained(
        "monologg/kobert", cache_dir=train_args.cache_dir, config=config
    )

    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=train_args.cache_dir)  # [TODO]: data이름도 별도 데이터로 변경
    NSMC_train = NSMC_datasets["train"].map(preprocess, num_proc=10)  # [TODO]: num_proc 별도 args로 변경
    NSMC_test = NSMC_datasets["test"].map(preprocess, num_proc=10)

    accuracy = datasets.load_metric("accuracy")  # [TODO]: evaluate로 변경
    callback_func = [WandbCallback] if os.getenv("WANDB_DISABLED") != "true" else None
    collator = BertCollator(tokenizer=bert_tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=train_args,
        compute_metrics=metrics,
        train_dataset=NSMC_train,
        eval_dataset=NSMC_test,
        data_collator=collator,
        callbacks=callback_func,
        tokenizer=bert_tokenizer,
    )
    # do train, do eval, do predict으로 변경
    trainer.train(ignore_keys_for_eval=["attentions"])


if "__main__" == __name__:
    # [TODO]: 경로 관련 설정 삭제, 환경 변수 launch.json, shell script로 변경
    os.environ["WANDB_PROJECT"] = "[study]attention_machanism"
    os.environ["CODE_DIR"] = r"/home/jsb193/workspace/[study]Attention/Attention_Mechanism"
    # os.environ["WANDB_DISABLED"] = "false"

    parser = HfArgumentParser(CustomTrainingArguments)

    main(parser)
