import datasets
import torch
import torch.nn as nn
import transformers
from kobert_transformers import get_kobert_model, get_distilkobert_model
from setproctitle import setproctitle


def main():
    
    my_cache_dir = r"/data/jsb193/[study]Attention/.cache"
    NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=my_cache_dir)

    NSMC_datasets["train"]


if "__main__" == __name__:
    setproctitle("[JP]study_for_attention")
    main()
