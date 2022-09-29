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

Parameters for the test_config file:
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

#### Pytorch, HuggingFace
try:
    import torch
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'transformers'])
    import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from transformers import AutoTokenizer
    #from datasets import load_dataset
except:
    import pip
    pip.main(['install', 'transformers'])
    #pip.main(['install', 'datasets'])
    from transformers import AutoTokenizer
# Functions for saving during training already exist
from transformers import AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict


#### From github
UTILS_PATH = "../utils"
sys.path.append(UTILS_PATH)

from dataloaders import _add_to_text, create_context, create_full_context, create_context_dataset_from_df, create_context_dataset, get_perplexity_encodings, create_data_modeltrain, _anytype_context
from entropy_computation import batch_predict_logits_rnn, batch_predict_logits_lm, compute_perplexity
from entropy_computation import sentence_predict, test_predict_entropy, batch_predict_entropy, results_to_df
from entropy_computation import pivot_results_df

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
    parser.add_argument("--conf_file", type=str, required=True, help="Path to the file containing testing arguments & data info")
    
    args = parser.parse_args()
    assert os.path.exists(args.conf_file)
    with open(args.conf_file, "r") as f:
        test_config = json.load(f)

    # Model
    args.model = test_config['Model']['model_name'] 
    args.tokenizer = args.model if test_config['Model'].get('tokenizer', None) is None else test_config['Model']['tokenizer']
    args.tok_maxlength = 1024 if test_config['Model'].get('tok_maxlength', None) is None else test_config['Model']['tok_maxlength']
    args.load_from_checkpoint = test_config['Model'].get('load_from_checkpoint', None) # path to checkpoint
    args.local_only = test_config['Model'].get('local_files_only', False)

    # Data
    args.language = test_config['Dataset']['language']
    args.data = test_config['Dataset']['path']
    args.corpus = test_config['Dataset']['corpus']
    args.column_option = test_config['Dataset']['column_option']
    # DataLoader
    args.dataloader_kwargs = test_config.get("DataLoader", {"sep_token": "eos", "max_length": 150})

    # Experiments to run
    args.experiments = test_config.get("Computation", ["full_context"])
    # possible keys: "progressive_entropy" (ie context 0 to 8), "full_context" (ie 0 and full_context)
    # each experiment will generate different files
    # which sep is loaded in DataLoader config

    return args

def apply_pred_steps(df3, dataset_a, model, tokenizer):
    """
    df3: test pd.DataFrame
    dataset_a: finalised Dataset structure
    """
    dataset_a.set_format(type='torch', columns=['input_ids', 'start_idx', 'attention_mask'])
    test_dataloader = DataLoader(dataset_a, collate_fn=data_collator, batch_size=1, worker_init_fn=SEED)
    sent_avg_logp, tokens_logp, sent_length, sentence_tokens = test_predict_entropy(model, test_dataloader, tokenizer, DEVICE, batch_predict_logits_lm)
    test_dataframe = results_to_df(df3, sent_avg_logp, tokens_logp, sent_length, sentence_tokens = sentence_tokens, out_file_name = None)
    return test_dataframe

#%% ------ LM Train
if __name__ == '__main__':
    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace()
    start = datetime.now()
    MODEL_NAME = args.model

    ###### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_cont_kwargs = {'truncation':True, 'padding':'max_length', 'max_length': args.tok_maxlength}
    # Adding tokens for padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token

    logging.info('Loading data...')
    ###### Training (?) data
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    df = pd.read_csv(args.data, keep_default_na=False, na_values=['']) # na values: 'nan' for french
    logging.warning(f"Dropping {df.text.isna().sum() + (df.text == '').sum()} rows")
    df = df[~(df.text.isna()) & (df.text != "")]

    # Cannot be done before defining tokenizer
    sep_adapt = {"space": " ", "eos": tokenizer.eos_token}
    sep = args.dataloader_kwargs["sep_token"]
    args.dataloader_kwargs["sep_token"] = sep if sep not in sep_adapt.keys() else sep_adapt[sep]
    #args.dataloader_kwargs["max_length"] = 150 if "max_length" not in args.dataloader_kwargs.keys() else args.dataloader_kwargs["max_length"]

    dl_function = create_context_dataset
    dataset_a, df3 = dl_function(df, tokenizer, **args.dataloader_kwargs, **args.column_option)

    ###### Model
    logging.info("Loading model")

    if args.local_only:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if args.load_from_checkpoint is not None:
        model, _, epoch, _ = checkpoint_load(args.load_from_checkpoint, model)
    model.to(DEVICE) # loading model

    # saving df
    ptest_dataframe = []
    out_file_name = f'{'.'.join(args.data.split[:-1])}-{MODEL_NAME}.csv'
    ###### Computing entropy
    if "progressive_entropy" in args.experiments:
        logging.info(f"Computing entropy with context 1 to 8 sentences, using model")

        for i in range(1,8):
            logging.info(f'\n------------- CONTEXT {i} -------------')
            df3['context'] = create_context(df3, context_len = i, file_col = file_col) # from parameters at top

            # Compute
            dataset_c = create_context_dataset_from_df(df, tokenizer, f"context", files_train, files_test, batch_size=BATCH_SIZE, max_length=1024)            
            test_dataframe = apply_pred_steps(df3, dataset_c, model, tokenizer)
            test_dataframe['mode'] = f'context-{i}'
            ptest_dataframe.append(test_dataframe.copy())


    if "full_context" in args.experiments:
        logging.info(f"Computing entropy with no and full context, using model")
        # no need to add for loops for the various configurations, can be done by calling the script again with new config.

        # With context
        logging.info(f'\n------------- PARAM {args.dataloader_kwargs} -------------')

        #dataset_a, df3 = dl_function(df, tokenizer, **args.dataloader_kwargs, **args.column_option)
        test_dataframe = apply_pred_steps(df3, dataset_a, model, tokenizer)
        test_dataframe['mode'] = 'full-context'

        ptest_dataframe.append(test_dataframe.copy())
        # in case of crash
        #pd.concat(ptest_dataframe, axis=0, ignore_index=True)[[col for col in test_dataframe if 
        #        col not in ['text_input_ids_full', 'text_input_ids', 'input_ids', 'attention_mask']]].to_csv(
        #    out_file_name,index=False)

    # No context outside of loop, always returned
    logging.info(f'\n------------- PARAM NO CONTEXT -------------')
    # only need to recreate dataset, not df, since using eos parametrized
    df3['input_ids'] = df3['text_input_ids'] # no context
    df3['start_idx'] = df3.apply(lambda x: len(x.input_ids) - x.length, axis = 1)
    df3['attention_mask'] = df3.input_ids.apply(lambda x: [1]*len(x))
    dataset_a = Dataset.from_pandas(df3)
    #dataset_a.set_format(type='torch', columns=['input_ids', 'start_idx', 'attention_mask'])
    #test_dataloader = DataLoader(dataset_a, collate_fn=data_collator, batch_size=1, worker_init_fn=SEED)
    #sent_avg_logp, tokens_logp, sent_length, sentence_tokens = test_predict_entropy(model, test_dataloader, tokenizer, DEVICE, batch_predict_logits_lm)
    #test_dataframe = results_to_df(df3, sent_avg_logp, tokens_logp, sent_length, sentence_tokens = sentence_tokens, out_file_name = None)
    test_dataframe = apply_pred_steps(df3, dataset_a, model, tokenizer)

    test_dataframe['mode'] = 'no-context'
    ptest_dataframe.append(test_dataframe.copy())

    # Output
    logging.info(f'Saving dataframe to {out_file_name}')
    test_dataframe['model'] = MODEL_NAME
    ptest_dataframe = pd.concat(ptest_dataframe, axis=0, ignore_index=True)
    ptest_dataframe = ptest_dataframe[[col for col in ptest_dataframe if col not in ['text_input_ids_full', 'text_input_ids', 'input_ids', 'attention_mask']]]
    ptest_dataframe.to_csv(out_file_name,index=False)
