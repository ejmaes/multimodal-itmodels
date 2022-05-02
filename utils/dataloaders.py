from multiprocessing.sharedctypes import Value
import os
import pandas as pd
import numpy as np
from datetime import datetime
import itertools

from sklearn.model_selection import train_test_split
from transformers.data.data_collator import default_data_collator
from transformers import TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict # also huggingface

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



#%% Create context & add sep tokens
def _add_to_text(df:pd.DataFrame, text_col:str='text',
                   speaker_col:str='speaker', file_col:str='file',
                   add_ipu_speaker_tokens:bool=False, add_speaker_tokens:bool=False, **kwargs):
    """Common to create_context functions - update text with needed speaker tags

    Input
    ----------
    add_speaker_tokens: bool, default False,
        Whether to add the speaker tokens everytime there is a change

    add_ipu_speaker_tokens: bool, default False
        Whether to add the speaker tokens _for each ipu_; if True then add_speaker_token is also technically applied
    
    Returns
    -----------
    updated_text: pd.Series
        The updated column containing the text
    """
    if add_ipu_speaker_tokens:
        updated_text = df.apply(lambda x: f"<{x[speaker_col]}> {x[text_col]}", axis=1)
    elif add_speaker_tokens:
        df['cs'] = ((df[file_col] != df[file_col].shift()) | (df[speaker_col] != df[speaker_col].shift()))
        # Different ways to do it: columns multiplication cs*speaker + text - but will end up adding a .apply anyway to add the <> to the token
        updated_text = df.apply(lambda x: f"<{x[speaker_col]}> {x[text_col]}" if x.cs else x[text_col], axis=1)
    else: # in order not to rewrite text column (case of multiple experiments)
        updated_text = df[text_col]

    return updated_text

def create_context(df: pd.DataFrame, context_len:int=5, text_col:str='text',
                   speaker_col:str='speaker', file_col:str='file', 
                   sep_token:str=' ', sep_context_sent:bool=False, **text_kwargs) -> pd.Series:
    """Creates a context based on a fixed number of IPUs anterior to the sentence, and concat, eventually using tokens/speakers to separate

    Ways to adapt:
        * Add a sep_token _before_ starting the sentence
        * Add a sep_token _after_ finishing the sentence

    Input
    --------
    sep_token: str, default None
        Which token to add at the end of the sentence (tokenizer.eof_token, tokenizer.sep_token...)

    sep_context_sent: bool, default False
        Whether to add a separator between context and text. If True, sep_token is only applied there; if False, sep_token is applied everywhere
    """
    df[f'{text_col}_u'] = _add_to_text(df, text_col=text_col, speaker_col=speaker_col, file_col=file_col, **text_kwargs)
    join_sep = ' ' if sep_context_sent else sep_token 
    text_col = f'{text_col}_u' # for further usage

    # naming columns f'shift_-{str(i).zfill(1+context_len//10)}'
    prev_sentences = pd.concat([df[text_col].shift(-x) for x in range(-context_len,0)], 
                               axis=1, keys=[f'shift_{i}' for i in range(-context_len,0)])
    prev_files = pd.concat([df[file_col] == df[file_col].shift(-x) for x in range(-context_len,0)], 
                           axis=1, keys=[f'shift_{i}' for i in range(-context_len,0)])
    prev_sentences = prev_sentences*prev_files # removing context obtained from previous files
    prev_sentences.fillna('', inplace=True)
    # columns are (normally) ordered to be joined correctly
    sentence_context = prev_sentences.apply(join_sep.join, axis=1)
    # add context to text, separated by (eventual) sep_token (default space) and return
    cc = (sentence_context+sep_token+df[text_col]).apply(lambda x: x.strip().replace('  ',' ')) 

    return cc

def create_full_context(df: pd.DataFrame, text_col:str='text',
                   speaker_col:str='speaker', file_col:str='file', index_col:str='index',
                   sep_token:str=' ', sep_context_sent:bool=False, **text_kwargs) -> pd.Series:
    """Creates a context containing every IPU previously used, and concat, eventually using tokens/speakers to separate

    Ways to adapt:
        * Add a sep_token _before_ starting the sentence
        * Add a sep_token _after_ finishing the sentence

    Input
    --------
    sep_token: str, default None
        Which token to add at the end of the sentence (tokenizer.eof_token, tokenizer.sep_token...)

    sep_context_sent: bool, default False
        Whether to add a separator between context and text. If True, sep_token is only applied there; if False, sep_token is applied everywhere
    """
    df[f'{text_col}_u'] = _add_to_text(df, text_col=text_col, speaker_col=speaker_col, file_col=file_col, **text_kwargs)
    join_sep = ' ' if sep_context_sent else sep_token 
    text_col = f'{text_col}_u' # for further usage
    
    # Method 1 - slightly slower
    #c = df.groupby(file_col).agg({text_col:list})[text_col].to_dict()
    #df['text_full'] = df.apply(lambda x: ' '.join(c[x[file_col]][:x[index_col]+1]).strip(), axis=1)
    # Method 2 - faster
    """
    c = df.groupby(file_col)
    cc = pd.DataFrame(c.agg({text_col:list})[text_col].apply(lambda x: [join_sep.join(x[:i]) for i in range(len(x))]).explode() # note: text will not be used since added later
                        ).reset_index(drop=False)
    cc[index_col] = c.agg({text_col:len})[text_col].apply(lambda x: range(x)).explode().reset_index(drop=True)
    df = pd.merge(left=df, left_on=[file_col, index_col], right=cc, right_on=[file_col, index_col], suffixes=('','_full')) # merging bc index might not be ordered identically bc of groupby
    """
    c = df.groupby(file_col).agg({
        text_col: lambda x: [join_sep.join(list(x)[:i]) for i in range(len(x))], # if using +1 then not adding the last line (concat)
        index_col: list
    })
    c = c.explode([text_col, index_col]).reset_index(drop=False)
    df = pd.merge(left=df, left_on=[file_col, index_col], right=c, right_on=[file_col, index_col], how='outer', suffixes=('','_full')) # merging bc index might not be ordered identically bc of groupby
    # full_text is currently only context, adding text and separator:
    df[f'{text_col}_full'] = (df[f'{text_col}_full']+sep_token+df[text_col]).apply(lambda x: x.strip().replace('  ',' ')) 

    return df[f'{text_col}_full']



#%% Create dataset
"""Dataset notes from https://huggingface.co/docs/datasets/use_dataset
```
padding (bool, str or PaddingStrategy, optional, defaults to False) –
Activates and controls padding. Accepts the following values:
* True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
* 'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
* False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths)
```
"""
def create_context_dataset_from_df(df: pd.DataFrame, tokenizer, context_col:str=None, 
                                files_train=None, files_test=None,
                                text_col:str='text', file_col:str='file', 
                                max_length:int=256, batch_size:int=8
                    ):
    """Create context dataset based on previously created column in dataframe
    """
    tok_cont_kwargs = {'truncation':True, 'padding':'longest', 'max_length':max_length}
    tok_text_kwargs = {'truncation':True, 'padding':False} if context_col is not None else tok_cont_kwargs

    # Dataset creation
    if (files_train is None) and (files_test is None):
        dataset_c = Dataset.from_pandas(df)
    else:
        dataset_c = DatasetDict({
            'train': Dataset.from_pandas(df[df[file_col].isin(files_train)]),
            'test': Dataset.from_pandas(df[df[file_col].isin(files_test)])
        })
    # First map text 
    dataset_c = dataset_c.map(lambda x: tokenizer(x[text_col], **tok_text_kwargs), batched=True, batch_size=batch_size)
    # Then get lengths & number of tokens which aren't padding
    #dataset_c = dataset_c.map(lambda x: {"length": x['attention_mask'].sum().item()}) # non 0 tokens are words
    dataset_c = dataset_c.map(lambda x: {"length": sum(x['attention_mask'])}) # non 0 tokens are words

    if context_col is not None: # if context included
        # Finally pad context
        dataset_c = dataset_c.map(lambda x: tokenizer(x[context_col], **tok_cont_kwargs),
                            batched=True, batch_size=batch_size)
        # Compute 'start_idx'
        dataset_c = dataset_c.map(lambda x: {"ct_length": sum(x['attention_mask'])}) # non 0 tokens are words
        dataset_c = dataset_c.map(lambda x: {"start_idx": x['ct_length'] - x['length']}) # non 0 tokens are words
    else:
        dataset_c = dataset_c.map(lambda x: {"start_idx": 0.}) # non 0 tokens are words

    return dataset_c


def create_context_dataset(dataframe: pd.DataFrame, tokenizer, files_train:list=None, files_test:list=None, 
                              text_col:str='text', file_col:str='file', index_col:str='index', speaker_col='speaker',
                              max_length:int=1024, batch_size:int=8,        
                              sep_token:str=' ', sep_context_sent:bool=False, **text_kwargs):
    """All-in-one function: Create context and dataset all at once dataset
    """
    # Copying df to avoid issues rerunning the function
    df = dataframe.copy(deep=True)

    # Parametrize
    df[f'{text_col}_u'] = _add_to_text(df, text_col=text_col, speaker_col=speaker_col, file_col=file_col, **text_kwargs)
    join_sep = ' ' if sep_context_sent else sep_token 
    text_col = f'{text_col}_u' # for further usage
    # Same but with tokenizes id
    sep_tok = tokenizer.encode(sep_token) if sep_token != ' ' else []
    join_st = sep_tok if not sep_context_sent else []
    rev_join_st = sep_tok if sep_context_sent else [] # ie, if sep_tok has been added to the sentences previously, no need to add it for final concat

    df['text_input_ids'] = df[text_col].apply(lambda x: join_st + tokenizer(x, truncation=True, padding=False)['input_ids'])
    df['length'] = df.text_input_ids.apply(len)

    #def prepend_ll(x:list, join_pat:list):
    #    """Add this pattern to each list element"""
    #    return [join_pat + y for y in x]

    c = df.groupby(file_col).agg({
        text_col: lambda x: [join_sep.join(list(x)[:i]) for i in range(len(x))], # if using +1 then not adding the last line (concat)
        'text_input_ids': lambda x: [(list(itertools.chain(*list(x)[:i]))) for i in range(len(x))], # if using +1 then not adding the last line (concat)
        index_col: list
    })
    c = c.explode([text_col, index_col, 'text_input_ids']).reset_index(drop=False)
    df = pd.merge(left=df, left_on=[file_col, index_col], right=c, right_on=[file_col, index_col], how='outer', suffixes=('','_full')) # merging bc index might not be ordered identically bc of groupby
    # full_text is currently only context, adding text and separator:
    df[f'{text_col}_full'] = (df[f'{text_col}_full']+sep_token+df[text_col])#.apply(lambda x: x.strip().replace('  ',' ')) 
    df['input_ids'] = df.apply(lambda x: (x['text_input_ids_full']+rev_join_st+x['text_input_ids'])[-max_length:], axis=1) # keep only the correct number of elements

    df['start_idx'] = df.apply(lambda x: len(x.input_ids) - x.length, axis = 1)
    df['attention_mask'] = df.input_ids.apply(lambda x: [1]*len(x))
    # Dataset 
    if files_train is not None:
        dataset_c = DatasetDict({
            'train': Dataset.from_pandas(df[df[file_col].isin(files_train)]),
            'test': Dataset.from_pandas(df[df[file_col].isin(files_test)])
        }) 
    else:
        dataset_c = Dataset.from_pandas(df)
      
    # drop extra columns from dataframe
    #df.drop(['text_input_ids', 'text_input_ids_full', 'attention_mask'], axis=1, inplace = True)
    return dataset_c, df


def get_perplexity_encodings(df:pd.DataFrame, tokenizer, 
                        text_col:str='text', file_col:str='file', 
                        sep_token:str=' ', files_test:list=None):
    """From HuggingFace documentation - https://huggingface.co/docs/transformers/perplexity
    Adding sep parameter to match data used in experiments (can be ' ' or tokenizer.eos_token or...)
    """
    file_texts = df.groupby(file_col).agg({text_col: lambda x: sep_token.join(x).strip().replace('  ',' ')}).reset_index()
    if files_test is None:
        files_test = file_texts[file_col].unique()
    encodings = tokenizer("\n\n".join(file_texts[file_texts[file_col].isin(files_test)][text_col].tolist()), return_tensors = "pt")
    return encodings 

def create_data_modeltrain(df:pd.DataFrame, tokenizer, sep:str=' ', max_length:int=128, 
                           text_col:str='text',file_col:str='file', train_files:list=None, batch_size:int=8, **kwargs):
    """Similar to get_perplexity_encoding function, but more general (using train/test etc)
    Kwargs unused but might be used in place of create_context_dataset
    """
    # could be using function for encoding from perplexity
    sent_len = max_length // 2
    tmp = df.groupby('file').agg({'text': lambda x: sep.join(x).replace('  ', ' ').split(' ')})
    tmp.text = tmp.text.apply(lambda x: [' '.join(x[i-sent_len:i+sent_len]) for i in range(sent_len,len(x)-sent_len, sent_len)])
    tmp = tmp.explode('text').reset_index(drop=False)
    tmp = tmp[tmp['text'].apply(lambda x: isinstance(x,str))]

    if train_files is None: # taking random sentences
        tmp['tt'] = np.random.choice(2,tmp.shape[0],p=[0.7, 0.3])
        dataset_t = DatasetDict({
                    'train': Dataset.from_pandas(tmp[tmp.tt == 0]),
                    'test': Dataset.from_pandas(tmp[tmp.tt == 1])
                }) 
    else:
        dataset_t = DatasetDict({
                    'train': Dataset.from_pandas(tmp[tmp[file_col].isin(train_files)]),
                    'test': Dataset.from_pandas(tmp[~tmp[file_col].isin(train_files)])
                }) 

    dataset_t = dataset_t.map(lambda x: tokenizer(x['text'], truncation=True, padding='longest', max_length=sent_len), 
                              batched=True, batch_size=batch_size)
    return dataset_t