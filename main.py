import datasets
from kobert_tokenizer import KoBertTokenizer
from transformers import HfArgumentParser, Trainer, BertForSequenceClassification, BatchEncoding
from check_model import CheckForBertForSequenceClassification
from argument import StudyForTrainingArguments
from setproctitle import setproctitle
from integrations import CustomWansbCallBack
from collator import BertCollator
import os


def main(args):

    my_cache_dir = r"/data/jsb193/[study]Attention/.cache"

    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=my_cache_dir)
    bert_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", cache_dir=my_cache_dir)
    model = CheckForBertForSequenceClassification.from_pretrained(
        "monologg/kobert", cache_dir=my_cache_dir
    )

    def preprocess(input_data: datasets.Dataset) -> BatchEncoding:
        input_text = input_data["document"]
        output_data = bert_tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

    collator = BertCollator(tokenizer=bert_tokenizer)

    NSMC_train = NSMC_datasets["train"].map(preprocess, num_proc=10)
    NSMC_test = NSMC_datasets["test"].map(preprocess, num_proc=10)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=NSMC_train,
        eval_dataset=NSMC_test,
        data_collator=collator,
        callbacks=[CustomWansbCallBack],
    )

    trainer.train()


if "__main__" == __name__:
    os.environ["WANDB_PROJECT"] = "[study]attention_machanism"
    os.environ["CODE_DIR"] = r"/home/jsb193/workspace/[study]Attention/Attention_Mechanism"
    setproctitle("[JP]study_for_attention")
    parser = HfArgumentParser(StudyForTrainingArguments)
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    main(training_args)
