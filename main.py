import os
from typing import Dict, List, Union

import datasets
import torch
from setproctitle import setproctitle
from transformers import BatchEncoding, BertConfig, BertForSequenceClassification, HfArgumentParser

from argument import CustomTrainingArguments
from collator import BertCollator
from integrations import CustomWansbCallBack
from kobert_tokenizer import KoBertTokenizer
from trainer import CustomTrainer


def main(args: CustomTrainingArguments) -> None:

    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=args.cache_dir)
    bert_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", cache_dir=args.cache_dir)

    config = BertConfig(vocab_size=bert_tokenizer.vocab_size, num_labels=1)
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", cache_dir=args.cache_dir, config=config)
    model = model.to("cpu")

    def preprocess(input_data: datasets.Dataset) -> BatchEncoding:
        input_text = input_data["document"]
        output_data = bert_tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

    collator = BertCollator(tokenizer=bert_tokenizer)

    NSMC_train = NSMC_datasets["train"].map(preprocess, num_proc=10)
    NSMC_test = NSMC_datasets["test"].map(preprocess, num_proc=10)

    callback_func = [CustomWansbCallBack] if os.getenv("WANDB_DISABLED") != "true" else None
    accuracy = datasets.load_metric("accuracy")

    def metrics(outputs: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, int]:
        predictions = outputs.predictions
        references = outputs.label_ids
        result = accuracy._compute(predictions, references, normalize=False)
        return {"Accuracy": result}

    trainer = CustomTrainer(
        model=model,
        args=args,
        compute_metrics=metrics,
        train_dataset=NSMC_train,
        eval_dataset=NSMC_test,
        data_collator=collator,
        callbacks=callback_func,
        tokenizer=bert_tokenizer,
    )

    trainer.train()


if "__main__" == __name__:
    os.environ["WANDB_PROJECT"] = "[study]attention_machanism"
    os.environ["CODE_DIR"] = r"/home/jsb193/workspace/[study]Attention/Attention_Mechanism"
    os.environ["WANDB_DISABLED"] = "false"
    setproctitle("[JP]test")

    parser = HfArgumentParser(CustomTrainingArguments)
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args)
