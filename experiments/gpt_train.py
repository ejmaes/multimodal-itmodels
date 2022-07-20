"""
1. Install libs
    pip install transformers
    pip install datasets
    pip install huggingface_hub

2. Check git lfs is installed (pushing on huggingface)
    apt-get install git-lfs

3. Configure github and create / clone huggingface models
    git config --global user.email "[user.email]"
    huggingface-cli login 
    git clone https://[user]:[token]@huggingface.co/[modelpath]

4. Check library / data is properly installed (paths are okay)
    rm -rf multimodal-itmodels
    git clone https://[gkey]@github.com/Neako/multimodal-itmodels.git

"""

import numpy as np
import os, sys
import pandas as pd
import time
import json
from tqdm import tqdm
from datetime import datetime

import argparse
from types import SimpleNamespace
import logging

#### Sklearn
from sklearn.model_selection import train_test_split

#### Pytorch, HuggingFace
need_install = []
try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
except:
    need_install.append('pytorch')
try:
    from transformers import AutoTokenizer
    # Functions for saving during training already exist
    from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
    from transformers import TextDataset, DataCollatorForLanguageModeling
except:
    need_install.append('transformers')

#### From github
UTILS_PATH = "../utils"
sys.path.append(UTILS_PATH)

from dataloaders import _add_to_text, create_context, create_full_context, create_context_dataset_from_df, create_context_dataset, get_perplexity_encodings, _anytype_context
from entropy_computation import sentence_predict, test_predict_entropy, batch_predict_entropy, results_to_df
from entropy_computation import batch_predict_logits_rnn, batch_predict_logits_lm, compute_perplexity
from entropy_computation import pivot_results_df



#%% ----- Parameters
DATA_PATH = "../data"
SAVE_PATH = '../models'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

with open("accounts_params.json", "r") as f:
    f = json.load(f)
HUB_PARAMS = f['huggingface'] # keys: user, token

#%% ------ Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Which model to load from HuggingFace.")
    parser.add_argument("--tokenizer", "-t", type=str, default=None, help="Name of the tokenizer to load from HuggingFace, if different from model name.") 
    # TODO: add mode to train tokenizer from BPE
    parser.add_argument("--conf_file", type=str, required=True, help="Path to the file containing training arguments & data info")
    parser.add_argument("--dataloader", "-d", type=str, default="gpt2", choices=["auto","custom"], help="Whether to use a custom DataLoader or the existing one for training.") 
    
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    with open(args.conf_file, "r") as f:
        training_config = json.load(f)
        args.language = training_config['Dataset']['language']
        args.data = training_config['Dataset']['path']
        args.column_option = training_config['Dataset']['column_option']
        args.traintest = None if 'TrainTest' not in training_config['Dataset'].keys() else training_config['Dataset']['TrainTest']
        args.training_kwargs = training_config['Trainer']

    return args

#%% ------ LM Train
if __name__ == '__main__':
    if len(need_install) > 0:
        raise ModuleNotFoundError(f"Missing packages: {', '.join(need_install)}")

    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace(model_type="gru", data_path="data/corlec", batch_size=8, nb_epochs=10, tokenizer="gpt2")
    start = datetime.now()
    MODEL_NAME = f"{args.language}_{args.corpus}"
    output_dir = os.path.join(SAVE_PATH, f"{MODEL_NAME}_{start.strftime('%Y%m%d-%H:%M:%S')}")

    ###### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Adding tokens for padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token

    ###### Training (?) data
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    df = pd.read_csv(args.data, keep_default_na=False, na_values=['']) # na values: 'nan' for french
    # train/test
    if args.traintest is not None:
        # for now only dealing with _one column_ for the split: len(args.column_option["group_columns"]) == 1
        args.file_column = args.column_option["group_columns"][0]
        if "train_groups" in args.traintest.keys(): # groups are specified
            files_train = args.traintest["train_groups"]
            files_test = args.traintest["test_groups"]
        else:
            #if len(args.column_option["group_columns"]) == 1:
            files = df[args.file_column].unique()
            files_train, files_test = train_test_split(files, random_state=SEED, test_size=0.3)

    dataset_c, df2 = create_context_dataset(df, tokenizer, files_train, files_test, sep_token=tokenizer.eos_token, max_length=150)

    ###### Model and Training
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    # initialise the TrainingArguments

    training_args = TrainingArguments(
        output_dir = f"./{MODEL_NAME}",
        overwrite_output_dir=True,
        # hub param
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_strategy="checkpoint",
        hub_token = HUB_PARAMS['token'],
        # batch, steps, eval & save params loaded from file
        evaluation_strategy="steps",
        **args.training_kwargs
    )
    # Train model
    trainer = Trainer(
        model=model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset_c['train_dataset'],
        eval_dataset = dataset_c['test_dataset']
    )
    trainer.train()
    trainer.save_model() # model will be saved in the training folder
    # lm = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True) # loading model
    trainer.push_to_hub(MODEL_NAME) 