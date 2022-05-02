from multiprocessing.sharedctypes import Value
import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from transformers.data.data_collator import default_data_collator
from transformers import TextDataset, DataCollatorForLanguageModeling

import torch
from torch.utils.data import DataLoader

#%% Create text files from general purpose CSV as df
def load_corpus(file_path:str, **kwargs):
    """Columns must be: file, index, speaker, text, (start, stop) - last two are optional 
    """
    if (isinstance(file_path, str)) and ('.csv' in file_path) and (os.path.exists(file_path)):
        df = pd.read_csv(file_path, **kwargs)
    # TODO: add checks on columns names
    print(df.columns)
    return df

def write_txt(lines:list, filepath:str):
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

def aggregate_dialog(df:pd.DataFrame, add_speaker_tokens:bool=False, add_ipu_speaker_tokens:bool=False) -> pd.DataFrame:
    """Returns a df containing, for each file, the aggregated text, with or without token markers
    """
    if add_speaker_tokens and not add_ipu_speaker_tokens:
        df['cumsum'] = ((df.file != df.file.shift()) | (df.speaker != df.speaker.shift())).astype(int).cumsum()
        df = df.groupby(['file','cumsum']).agg({
            'speaker': lambda x: list(x)[0],
            'text': lambda x: ' '.join(x)
        }).reset_index(drop=False)
        df['text'] = df.apply(lambda x: f"<{x.speaker}> {x.text}", axis=1)
    elif add_ipu_speaker_tokens:
        df['text'] = df.apply(lambda x: f"<{x.speaker}> {x.text}", axis=1)
    # then only concatenating wrt files
    return df.groupby('file').agg({ 'text': lambda x: ' '.join(x).strip().replace('  ',' ') })

def _create_context(df:pd.DataFrame, context_len:int=5, **kwargs) -> list:
    """Iterate on dataframe rows to get the context of N previous IPU
    """
    l = []
    for _, row in df.iterrows():
        # select the past 
        tmp = df[df.file == row.file]
        tmp = tmp[tmp['index'].between(max(0,row['index']-context_len), row['index'])]
        sentence_context = aggregate_dialog(tmp, **kwargs).text.iloc[0]
        l.append(sentence_context) 
    return l

def create_context(df: pd.DataFrame, context_len:int=5, add_ipu_speaker_tokens:bool=False) -> list:
    if add_ipu_speaker_tokens:
        df['text'] = df.apply(lambda x: f"<{x.speaker}> {x.text}", axis=1)
    # naming columns f'shift_-{str(i).zfill(1+context_len//10)}'
    prev_sentences = pd.concat([df.text.shift(-x) for x in range(-context_len,0)], 
                               axis=1, keys=[f'shift_{i}' for i in range(-context_len,0)])
    prev_files = pd.concat([df.file == df.file.shift(-x) for x in range(-context_len,0)], 
                           axis=1, keys=[f'shift_{i}' for i in range(-context_len,0)])
    prev_sentences = prev_sentences*prev_files # removing context obtained from previous files
    prev_sentences.fillna('', inplace=True)
    # columns are (normally) ordered to be joined correctly
    sentence_context = prev_sentences.apply(' '.join, axis=1).tolist()
    return [x.strip().replace('  ',' ') for x in sentence_context]


def tvt_split(text_list, ratio:list) -> dict:
    splits = {}
    s_ratio = 0.
    while len(ratio) >= 2:
        split_name = 'train' if len(splits) == 0 else f'test_{len(splits)-1}'
        r = ratio.pop(0)
        train, text_list = train_test_split(text_list, train_size=(r/(1-s_ratio)))
        splits[split_name] = train
        s_ratio += r
    splits['test'] = text_list
    return splits

def create_context_corpus(dialogs: pd.DataFrame, textdataset_path:str, ratio:list=[0.75, 0.25], 
                context_len:int=None, write_to_csv:bool=True,
                add_speaker_tokens:bool=False, add_ipu_speaker_tokens:bool=False):
    # step 1: concatenate each file
    dialogs['context_text'] = create_context(dialogs, context_len, 
                        add_speaker_tokens=add_speaker_tokens, add_ipu_speaker_tokens=add_ipu_speaker_tokens)
    # step 2: train/test split on the list
    splits = tvt_split(dialogs.index, ratio)
    if write_to_csv:
        for fname, lines in splits.items():
            name = f'{fname}_ctx-{context_len}_spk-{int(add_speaker_tokens)}{int(add_ipu_speaker_tokens)}.csv'
            dialogs.loc[lines].to_csv(os.path.join(textdataset_path, name), index=False)
    else:
        dialogs['split'] = None
        for split, lines in splits.items():
            dialogs.loc[lines, 'split'] = split
        return dialogs

def create_textdataset_corpus(dialogs:pd.DataFrame, textdataset_path:str, write_to_text:bool=True,
                ratio:list=[0.75, 0.25], add_speaker_tokens:bool=False, add_ipu_speaker_tokens:bool=False):
    """
    """
    # step 1: concatenate each file
    df = aggregate_dialog(dialogs, add_speaker_tokens, add_ipu_speaker_tokens)
    text_list = df.text.tolist()
    # step 2: train/test split on the list
    splits = tvt_split(text_list, ratio)
    # step 3: return or write to text
    if write_to_text:
        for fname, lines in splits.items():
            name = f'{fname}_spk-{int(add_speaker_tokens)}{int(add_ipu_speaker_tokens)}.txt'
            write_txt(lines, os.path.join(textdataset_path, name))
        with open(os.path.join(textdataset_path, "data_list.txt"), 'a+') as f:
            f.write(f"{datetime.now().strftime('%Y%m%d-%H:%M:%S')} - Creation of a text dataset WITH{'OUT'*(not add_speaker_tokens)} speaker tokens and WITH{'OUT'*(not add_ipu_speaker_tokens)} IPU tokens; ratios {ratio}; number of lines {[len(x) for x in splits.values()]}\n")
    else:
        return splits




#%% Existing DataLoaders
def load_dataset(files_path:list, tokenizer, block_size:int=8, stop_if_error:bool=True):
    """Creates a dataset from files with 1 line = 1 text / dialog (whole)

    Input:
    --------
    files_path: list of str
    block_size: int, number of tokens to have in one item
    stop_it_error: bool, whether to skip files with errors
    """
    datasets = {}
    for file_path in files_path:
        # check file path
        if not isinstance(file_path, str) or not os.path.exists(file_path) or not os.path.isfile(file_path):
            if stop_if_error:
                raise ValueError(f"'{file_path}' is not a correct file path")
            else: 
                continue

        # load dataset
        datasets[file_path.split('/')[-1]] = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size # block size is the max_length :| 
        )
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False) # causal here

    return datasets, data_collator


#%% Custom DataLoaders
class ContextualisedDatasetControl(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_seq_len, context_field, 
                df:pd.DataFrame=None, data_folder:str=None,
                add_special_tokens=True, context_length=0): # whether to add 
        if df is None and data_folder is None:
            raise ValueError("'df' and 'data_folder' parameters cannot both be empty")
        elif df is not None:
            self.data = df
        else: 
            self.data = pd.read_csv(data_folder) 
        
        self.add_special_tokens = add_special_tokens
        self.context_length = context_length
        self.text_column = '' if context_length > 0 else 'text'
        self.tokenizer = tokenizer

        if add_special_tokens:
            # add special tokens to tokenizer
            special_tokens = [f"<{x}>" for x in self.data.speaker.unique()]
            pass

        self.data['tokenized'] = self.data[self.text_column].apply(lambda x: self.tokenizer(x, return_tensors='pt'))

        """From Guilliani & Fernandez
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, 
                        add_special_tokens=False, max_length=max_seq_len + 2)
            self.data.append((inputs, idx))
        """
        self.model_data = {
            'inputs': None,
            'attention': None,
        }

        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.model_data[index]
