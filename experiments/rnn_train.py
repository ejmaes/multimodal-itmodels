from xml.dom import NotFoundErr
import numpy as np
import os, sys
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime

import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'transformers'])
    pip.main(['install', 'datasets'])
    from transformers import AutoTokenizer
    from transformers import DataCollatorForLanguageModeling
    from datasets import load_dataset, Dataset, DatasetDict

import argparse
from types import SimpleNamespace
import logging

UTILS_PATH = "../utils"
sys.path.append(UTILS_PATH)
from lstm import GRUModel
from dataloaders import create_context_dataset, get_perplexity_encodings

#%% Parameters
DATA_PATH = "./data"
SAVE_PATH = './models'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#%% Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--data_path", type=str, default=None, help="The path to the input pandas dataframe.")
    parser.add_argument("--dataenc_method", type=str, choices=['create_context_dataset', 'create_data_modeltrain'], help="Which function to use to create training data for the model.")
    parser.add_argument("--dataenc_args", type=dict, default={}, help="Specific parameters for the dataset creating function - text col, length of context, sep_token for instance.")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Fraction of data reserved for training.")
    parser.add_argument("--wikipedia_args", type=str, default=None, nargs=3, help="Pretraining on a wikipedia: tuple (language_dump, %train, %test).")
    # Model training args
    parser.add_argument("--model_savepath", type=str, required=True, help="Folder to save model to.")
    parser.add_argument("--model_path", type=str, default=None, help="Eventual location of a pretrained model to fine-tune.")
    parser.add_argument("--model_type", type=str, choices=["gru"], default="gru", help="Which type of model to train.")
    parser.add_argument("--is_checkpoint", action='store_true', help="Whether the pretrained model loaded is a checkpoint.")
    parser.add_argument("--save_checkpoints", type=int, default=-1, help="Frequency with which to save model checkpoints - default is -1 (don't save)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size.")
    parser.add_argument("--nb_epochs", "-e", type=int, default=10, help="Number of epochs to train the model on.")
    # Tokenizer args
    parser.add_argument("--tokenizer", "-t", type=str, default="gpt2", help="Name of the tokenizer to load from HuggingFace.") 
    parser.add_argument("--tok_maxlength", "-tml", type=int, default=1024, help="Max length (padding) for tokenizer to create examples.") 
    # TODO: add mode to train tokenizer from BPE

    # Also for dataframe: column settings
    dataframe_col_default = {'text_col':'text', 'file_col':'file', 'index_col':'index', 'theme_col':'theme', 'speaker_col':'speaker'}
    for k,v in dataframe_col_default.items():
        parser.add_argument(f"--{k}", type=str, default=v, help=f"Dataframe column for {k}.")

    args = parser.parse_args()
    if args.data_path is None and args.wikipedia_args is None:
        raise argparse.ArgumentError("Need at least one arguments between wikipedia_args and data_path")
    elif not os.path.exists(args.data_path):
        raise FileExistsError(f'Data file {args.data_path} not found.')
    args.dataenc_method = locals()[args.dataenc_method]

    return args

#%% RNN Train
def checkpoint_save(model, optimizer, epoch, loss, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path) # path must be .pt

def checkpoint_load(path:str, model, optimizer=None):
    """
    Input:
    -----------
    path: str
        path to the .pt checkpoint
    
    model: loaded model class with arguments
    optimizer: loaded optimizer class with arguments
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(model, train_dataloader, eval_dataloader,
          max_epoch:int=10, start_epoch:int=0, save_every:int=20,
          save_path:str=None,
          optimizer=None, criterion=nn.CrossEntropyLoss()):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())#, lr=0.001)
    if (save_path is not None) and (save_every > 0) and not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(start_epoch, max_epoch):
        logging.info(f"epoch {epoch}")
        model.train()
        train_loss = val_loss = 0
        for batch, data in tqdm(enumerate(train_dataloader)):
            x = data['input_ids'][:,:-1].to(DEVICE)
            y = data['input_ids'][:,1:].to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        for batch, data in tqdm(enumerate(eval_dataloader)):
            x = data['input_ids'][:,:-1].to(DEVICE)
            y = data['input_ids'][:,1:].to(DEVICE)
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred.transpose(1, 2), y)
                val_loss += loss.item()

        logging.info({ 'epoch': epoch, 'train_loss': train_loss, 'test_loss': val_loss })
        if (save_path is not None) & (save_every > 0) & ((epoch+1) % save_every == 0):
            # save current model
            checkpoint_save(model, optimizer, epoch, loss, os.path.join(save_path, f'model_epoch_{epoch+1}.pt'))


if __name__ == '__main__':
    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace(model_type="gru", data_path="data/corlec", batch_size=8, nb_epochs=10, tokenizer="gpt2")
    start = datetime.now()
    output_dir = os.path.join(SAVE_PATH, f"{args.model_type}_{args.corpus}_{start.strftime('%Y%m%d-%H:%M:%S')}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_cont_kwargs = {'truncation':True, 'padding':'max_length', 'max_length':args.tok_maxlength}

    logging.info('Loading data...')
    if args.wikipedia_args is not None:
        lg_dump = args.wikipedia_args[0]
        train_split = f'train[:{args.wikipedia_args[1]}%]'
        test_split = f'train[{args.wikipedia_args[1]}:{args.wikipedia_args[1]+args.wikipedia_args[2]}%]'
        try:
            dataset = DatasetDict({
                'train': load_dataset("wikipedia", lg_dump, split=train_split),
                'test': load_dataset("wikipedia", lg_dump, split=test_split)
            })
        except:
            raise NotFoundErr('examples: 20220301.fr, 20220301.en... - List of available dumps is available here: https://huggingface.co/datasets/wikipedia')
        dataset = dataset.map(lambda x: tokenizer(x['text'], **tok_cont_kwargs), batched=True, batch_size=args.batch_size)
    
    else:
        df = pd.read_csv(args.data_path, keep_default_na=False, na_values=[''])
        args.theme_col = args.theme_col if args.theme_col in df.columns else None

        if args.theme_col is not None:
            with_theme = shuffle(df[(~df[args.theme_col].isna()) & (df[args.theme_col] != '')][args.file_col].unique(), random_state=SEED)
            no_theme = shuffle(list(set(df[args.file_col].unique()) - set(with_theme)), random_state=SEED)
            # each list shuffled
            c = list(no_theme) + list(with_theme)
            n = int(len(c) * (1-args.train_fraction))
            files_train = c[:n]
            files_test = c[n:]
        else:
            files = df.file.unique()
            files_train, files_test = train_test_split(files, random_state=SEED, test_size=0.3)
        logging.info(f'Training files: \n{files_train} \n\n Testing files: \n{files_test}')

        dataset = args.dataenc_method(df, tokenizer, max_length=args.max_length, batch_size=args.batch_size, **args.dataenc_args)

    dataset.set_format(type='torch', columns=['input_ids','attention_mask'])
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    train_dataloader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=args.batch_size, worker_init_fn=SEED)
    test_dataloader = DataLoader(dataset['test'], collate_fn=data_collator, batch_size=args.batch_size, worker_init_fn=SEED)
    
    model = GRUModel(tokenizer)
    optimizer = optim.Adam(model.parameters())#, lr=0.001)
    if args.model_path is not None:
        logging.info(f"Loading existing model from {args.model_path}")
        if args.is_checkpoint:
            model, optimizer, epoch, loss = checkpoint_load(args.model_path, model, optimizer)
        else:
            model.load_state_dict(torch.load(args.model_path))
    logging.info(f"Training model for {args.nb_epochs} with bs {args.batch_size}")
    # Training
    train(model, train_dataloader, test_dataloader, max_epoch=args.nb_epochs, start_epoch=epoch, save_every=args.save_checkpoints, save_path=args.model_savepath, optimizer=optimizer)
    # Saving
    if args.model_savepath[-3:] != '.pt':
        args.model_savepath += '.pt'
    torch.save(model.state_dict(), args.model_savepath)
    logging.info(f"Model successfully saved to {args.model_savepath}")