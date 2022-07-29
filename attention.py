import torch
import torch.nn as nn
import transformers
import datasets


my_cache_dir = f"/data/jsb193/[study]Attention"
NSMC_datasets = datasets.load_dataset("nsmc", cache_dir=my_cache_dir)
