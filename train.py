import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from transformers import AutoTokenizer

import argparse
from types import SimpleNamespace

#%% Parameters
DATA_PATH = "./data"
SAVE_PATH = './models'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#%% Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["gru","gpt"], required=True, help="Which type of model to train.")
    parser.add_argument("--data_path", type=str, required=True, help="The path to the input pandas dataframe.")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size.")
    parser.add_argument("--nb_epochs", "-e", type=int, default=10, help="Number of epochs to train the model on.")
    parser.add_argument("--tokenizer", "-t", type=str, default="gpt2", help="Name of the tokenizer to load from HuggingFace.") 
    # TODO: add mode to train tokenizer from BPE
    parser.add_argument("--dataloader", "-d", type=str, default="gpt2", choices=["auto","custom"], help="Whether to use a custom DataLoader or the existing one for training.") 
    args = parser.parse_args()

    return args

#%% RNN Train
def checkpoint_save(model, optimizer, epoch, loss, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path) # path must be .pt

def checkpoint_load(path:str, model, optimizer):
    model = TheModelClass(*args, **kwargs)
    optimizer = TheOptimizerClass(*args, **kwargs)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

def train_gru(model, train_dataset):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())#, lr=0.001)
    model.train()

    for epoch in range(2):
        print(f"epoch {epoch}", end=" ")
        for batch, x in tqdm(enumerate(train_dataset)):
            x = x.unsqueeze(0).to(DEVICE) # because only one
            y = x[:,1:].to(DEVICE)
            x = x[:,:-1]

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            #print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


#%% LM Train
# Functions for saving during training already exist
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

from transformers.data.data_collator import default_data_collator
from transformers import TextDataset, DataCollatorForLanguageModeling


if __name__ == '__main__':
    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace(model_type="gru", data_path="data/corlec", batch_size=8, nb_epochs=10, tokenizer="gpt2")
    start = datetime.now()
    output_dir = os.path.join(SAVE_PATH, f"{args.model_type}_{args.corpus}_{start.strftime('%Y%m%d-%H:%M:%S')}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.dataloader == "auto":
        datasets, data_collator = load_dataset([], tokenizer, block_size=8)
    else:
        pass

    if args.model_type == "gpt":
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        # initialise the TrainingArguments
        training_args = TrainingArguments(
            output_dir = output_dir,
            #overwrite_output_dir=True, # no need since date
            #num_train_epochs = 2,
            max_steps=400,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            eval_steps=200,
            save_steps=400,
            warmup_steps=100,
            prediction_loss_only=False#True
        )
        # Train model
        trainer = Trainer(
            model=model,
            args = training_args,
            data_collator = data_collator,
            train_dataset = datasets['train_dataset'],
            eval_dataset = datasets['test_dataset']
        )
        trainer.train()
        trainer.save_model() # model will be saved in the training folder
        # lm = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True) # loading model
    
    elif args.model_type == "gru":
        model = 
