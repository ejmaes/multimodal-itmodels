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

Parameters for the training_config file:
    Model > model_name, tokenizer: paths from HuggingFace - TODO: parameter to create tokenizer from scratch
    Dataset > corpus, path, language; TrainTest > train_groups, test_groups if defined, otherwise column to split on
    Trainer: parameters from HuggingFace doc, to add to the Trainer (nb epochs, batch size...)
    DataLoader > context_max_length, separator (when collating sentences), loader_form (whether to use the same loader as in prediction - sentence by sentence: 'sent' - or collated - 'collate')
"""

import numpy as np
import os, sys
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime

import argparse
from types import SimpleNamespace
import logging
import random

#### Sklearn
from sklearn.model_selection import train_test_split

#### Pytorch, HuggingFace
try:
    import torch
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'transformers'])
    import torch
import torch.nn.functional as F
import torch.optim as optim
try:
    from transformers import AutoTokenizer
    #from datasets import load_dataset
except:
    import pip
    pip.main(['install', 'transformers'])
    #pip.main(['install', 'datasets'])
    from transformers import AutoTokenizer
# Functions for saving during training already exist
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling


#### From github
UTILS_PATH = "../utils"
sys.path.append(UTILS_PATH)

from dataloaders import _add_to_text, create_context, create_full_context, create_context_dataset_from_df, create_context_dataset, get_perplexity_encodings, create_data_modeltrain, _anytype_context
from entropy_computation import batch_predict_logits_rnn, batch_predict_logits_lm, compute_perplexity



#%% ----- Parameters
DATA_PATH = "../data"
SAVE_PATH = '../models'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open("accounts_params.json", "r") as f:
    f = json.load(f)
HUB_PARAMS = f['huggingface'] # keys: user, token

#%% ------ Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, required=True, help="Path to the file containing training arguments & data info")
    
    args = parser.parse_args()
    assert os.path.exists(args.conf_file)
    with open(args.conf_file, "r") as f:
        training_config = json.load(f)

    # Model
    args.model = training_config['Model']['model_name']
    args.tokenizer = args.model if training_config['Model'].get('tokenizer', None) is None else training_config['Model']['tokenizer']
    # TODO: add mode to train tokenizer from BPE
    args.tok_maxlength = 1024 if training_config['Model'].get('tok_maxlength', None) is None else training_config['Model']['tok_maxlength']

    # Data
    args.language = training_config['Dataset']['language']
    args.data = training_config['Dataset']['path']
    args.corpus = training_config['Dataset']['corpus']
    args.column_option = training_config['Dataset']['column_option']
    args.traintest = None if 'TrainTest' not in training_config['Dataset'].keys() else training_config['Dataset']['TrainTest']
    # Trainer / DataLoader
    args.training_kwargs = training_config['Trainer']
    args.dataloader = 'sent' if training_config["DataLoader"].get('loader_form', None) is None else training_config["DataLoader"].pop('loader_form')
    args.dataloader_kwargs = training_config.get("DataLoader", {"sep_token": "eos"})

    # Perplexity
    args.perplexity_kwargs = training_config.get("Perplexity", {"stride":16, "max_length":1024})

    return args

#%% ------ LM Train
if __name__ == '__main__':
    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace(model_type="gru", data_path="data/corlec", batch_size=8, nb_epochs=10, tokenizer="gpt2")
    start = datetime.now()
    MODEL_NAME = f"{args.language}_{args.corpus}_{args.model.replace('/','-')}"

    ###### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_cont_kwargs = {'truncation':True, 'padding':'max_length', 'max_length':args.tok_maxlength}
    # Adding tokens for padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token

    logging.info('Loading data...')
    ###### Training (?) data
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    df = pd.read_csv(args.data, keep_default_na=False, na_values=['']) # na values: 'nan' for french
    logging.warning(f"Dropping {df.text.isna().sum() + (df.text == '').sum()} rows")
    df = df[~(df.text.isna()) & (df.text != "")]
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
    logging.info(f'Training files: \n{files_train} \n\n Testing files: \n{files_test}')

    # Cannot be done before defining tokenizer
    sep_adapt = {"space": " ", "eos": tokenizer.eos_token}
    sep = args.dataloader_kwargs["sep_token"]
    args.dataloader_kwargs["sep_token"] = sep if sep not in sep_adapt.keys() else sep_adapt[sep]
    args.dataloader_kwargs["max_length"] = 150 if "max_length" not in args.dataloader_kwargs.keys() else args.dataloader_kwargs["max_length"]

    if args.dataloader == 'sent':
        dl_function = create_context_dataset
    elif args.dataloader == 'collate':
        dl_function = create_data_modeltrain
    else:
        raise ValueError('Undefined method for loading data')
    dataset_c, df2 = dl_function(df, tokenizer, files_train=files_train, files_test=files_test, 
                                **args.dataloader_kwargs, **args.column_option)

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
        train_dataset = dataset_c['train'],
        eval_dataset = dataset_c['test']
    )
    
    logging.info(f"Training model")
    trainer.train()
    trainer.save_model() # model will be saved in the training folder
    # lm = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True) # loading model
    trainer.push_to_hub(MODEL_NAME) 
    logging.info(f"Model saved to Hub! Total time elapsed: {datetime.now() - start}")

    # Perplexity:
    for sep in [' ', tokenizer.eos_token]:
        encodings = get_perplexity_encodings(df, tokenizer, sep_token = sep, files_test=files_test)
        ppl = compute_perplexity(model, encodings, DEVICE, model_is_lm=True, **args.perplexity_kwargs)
        logging.info(f"Model perplexity on the test dataset with separator `{sep}`: {ppl}")