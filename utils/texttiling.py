"""
# TextTiling application on dialogues

Sources on TextTiling: 
* https://stackoverflow.com/questions/25072167/split-text-into-paragraphs-nltk-usage-of-nltk-tokenize-texttiling
* https://github.com/stylianipantela/texttiling/blob/master/texttiling.py
* https://www.nltk.org/_modules/nltk/tokenize/texttiling.html
* https://github.com/levy5674/text-tiling-demo/blob/master/text_tiling_demo/demo.py

### Method for analysis: 
1. For one file: get all utterances & concat them in "paragraphs" (utterances must be separated by "\n\n")
2. Separate groups are in different _tokens_. Linking sentence to original row can be done using the "\n" tag to split the paragraph into separate utterances
3. Apply to all files

### Usage example:
```python
ttt = TextTilingTokenizer()
tokens = ttt.tokenize(text_blob)

for token in tokens:
    paragraph = token.replace("\n", " ")
    print(paragraph)
    print("\n") 
```
"""


import pandas as pd
import os, sys, re, glob
from tqdm import tqdm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

from nltk.tokenize import TextTilingTokenizer


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", '-d', type=str, required=True, help="The path to the input pandas dataframe.")
    parser.add_argument("--text_col", '-tc', type=str, default='text', help="The name of the column containing utterances.")
    parser.add_argument("--file_col", '-fc', type=str, default='file', help="The name of the column containing file references.")
    parser.add_argument("--image_comp", '-i', action='store_true', help="Whether to plot the comparison between original indexes and TextTiling.")
    parser.add_argument("--theme_index_col", '-tic', type=str, default=None, help="The name of the column containing themes if exists.")
    parser.add_argument("--file_index_col", '-fic', type=str, default=None, help="The name of the column containing file indexes in dataframe.")

    args = parser.parse_args()
    if args.image_comp and (args.theme_index_col is None or args.file_index_col is None):
        raise ValueError('`-tic` and `-fic` arguments need to be set.')

    return args

# %% TextTiling Application Functions
def files_to_paragraphs(df:pd.DataFrame, text_col:str="text", file_col:str="file") -> dict:
    """Generates a dictionary containing the 'paragraphs' to be analysed by TextTiling, associated with 
    their respective files
    """
    d = df.groupby(file_col).agg({text_col: lambda x: '\n\n'.join(x)})[text_col].to_dict()
    return d
    
def one_tt(s:str, ttt = TextTilingTokenizer()):
    """Analyse a conversation using TextTiling to generate topics segmentation.
    Then retrace to initial conversation using '\n' as utterance breaks.
    Returns a dict and a list: the dict contains a list of utterances associated with each topic, 
    the list containing the index of the topic for each utterance (shape: df[df.file == X].shape[0])
    """
    d_utter = {}
    l_idx = []
    tokens = ttt.tokenize(s)
    for i, token in enumerate(tokens):
        tok_utter = token.split("\n\n")
        tok_utter = [x for x in tok_utter if x != ''] # remove ' from list
        d_utter[i] = tok_utter
        l_idx += [i]*len(tok_utter)

    return d_utter, l_idx

def df_apply_tt(df:pd.DataFrame, text_col:str="text", file_col:str="file"):
    """Apply TextTiling to all files in the dataframe
    Return a list of theme index to be used as a df column, and two dictionaries containing results details
    """
    files_paragraph = files_to_paragraphs(df, text_col, file_col)
    all_utter = all_idx = {}
    for file, s in tqdm(files_paragraph.items()):
        d_utter, l_idx = one_tt(s)
        assert(df[df.file == file].shape[0] == len(l_idx))
        all_utter[file] = d_utter # json
        all_idx[file] = l_idx

    # not working order issues - getting indexes directly with dataframe file order
    #l_idx = list(itertools.chain(*all_idx.values()))
    l_idx = list(itertools.chain( *[all_idx[f] for f in df[file_col].drop_duplicates().to_list()]))
    return l_idx, all_utter, all_idx


#%% Comparison of results
def plot_comparison(df, file_col, file_index_col, theme_index_col, output_path):
    # Pivot dataframe
    dfm = df[[file_col, file_index_col, theme_index_col, 'tt_index']].melt([file_col,file_index_col], 
                                        var_name='theme_idx_origin', value_name='value')
    g = sns.relplot(
        data=dfm,
        x=file_index_col, y="value", 
        hue="theme_idx_origin", col=file_col, col_wrap=5, kind="line", 
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )
    # Save to file # PairPlot using 'fig' but other plots might use 'figure'
    getattr(g, 'fig').savefig(output_path)

def save_df(df:pd.DataFrame, df_out_path):
    # check if file already exists:
    files_present = glob.glob(df_out_path)
    if files_present:
        print('Replacing file...')
        os.remove(df_out_path)
    df.to_csv(df_out_path, index=False)


#%% Update file if no theme index
def update_theme_index(df:pd.DataFrame, file_col:str, theme_col:str):
    # get a dictionary with each theme indexed 
    # first apply gets unique themes, second associates index with each
    # (two steps needed otherwise all files have the same dict length)
    d = df[[file_col,theme_col]].dropna().groupby(file_col)[theme_col].apply(lambda x: x.unique()
                ).apply(lambda x: {y:i for i,y in enumerate(x)}).to_dict()
    l = df.apply(lambda x: x[theme_col] if str(x[theme_col]) == 'nan' else d[x[file_col]][x[theme_col]], axis=1).fillna(-1)
    return l


#%% Main
if __name__ == '__main__':
    args = arg_parser()
    df = pd.read_csv(args.data_path, keep_default_na=False, na_values=['']) # for paco-cheese

    l_idx, all_utter, all_idx = df_apply_tt(df, args.text_col, args.file_col)
    df['tt_index'] = l_idx
    df_out_path = args.data_path.replace('.csv', '_texttiling.csv')
    save_df(df, df_out_path)

    if args.image_comp:
        if args.theme_index_col not in df.columns:
            raise ValueError(f'`-tic` argument needs to be set to an existing column of the dataframe - one of {df.columns}.')
        elif args.file_index_col not in df.columns:
            raise ValueError(f'`-fic` argument needs to be set to an existing column of the dataframe - one of {df.columns}.')
        elif df[args.theme_index_col].dtype == 'O':
            df['theme_idx'] = update_theme_index(df, args.file_col, args.theme_index_col)
            args.theme_index_col = 'theme_idx'
            # update df
            save_df(df, df_out_path)
        image_path = args.data_path.replace('.csv', '_texttiling_comparison.png')
        plot_comparison(df, args.file_col, args.file_index_col, args.theme_index_col, image_path)


