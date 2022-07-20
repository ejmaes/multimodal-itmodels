import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os, sys, re
from glob import glob
import subprocess
from tqdm import tqdm

from collections import Counter
from itertools import chain
from functools import partial
from difflib import SequenceMatcher

from scipy.signal import detrend
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import cohen_kappa_score
from ast import literal_eval

#%% Functions for removing start of dialogue

def remove_jokes(df):
    # if has theme
    # locate end of intro themes
    themes = ['hetero selection', 'frog joke', 'donkey joke']
    file_drop_till = df.groupby(['file','theme'])['index'].max().reset_index(drop=False)
    file_drop_till = file_drop_till[file_drop_till.theme.apply(lambda x: isinstance(x,str) and any([y in x for y in themes]))]
    file_drop_till = file_drop_till.groupby('file')['index'].max().to_dict()
    # remove those index from conversation
    for file, index_min in file_drop_till.items():
        df = df[~((df.file == file) & (df['index'] <= index_min))]
    
    return df

def apply_removal(dataframe:pd.DataFrame):
    themed_files = dataframe.drop_duplicates(subset=['file','has_theme']).file.tolist()
    df = dataframe[dataframe.has_theme]
    df = remove_jokes(df)
    df['index'] = df.groupby(['model','file']).agg({'text':lambda x: range(len(x))}).explode('text')['text'].tolist()
    return df, themed_files


#%% Functions for inverting theme indexing

def invert_theme_index(df:pd.DataFrame):
    d2 = df.copy(deep=True)
    d2['theme_index'] = (d2.theme != d2.theme.shift()).cumsum()
    tig = d2.groupby(['corpus','file','dyad', 'theme_index'])
    for _, group in tig:
        idx = group.index
        d2.loc[idx,'theme_index'] = list(range(-len(idx), 0))

    df['theme_index_inv'] = d2['theme_index']
    df['theme_index_invpos'] = df.theme_index_inv.abs()
    df['distance_to_theme_break'] = df[['theme_index_invpos','theme_index']].min(axis=1)
    df['sign_to_theme_break'] = df[['theme_index_invpos','theme_index']].idxmin(axis=1)
    df['na_distance_to_theme_break'] = df.apply(lambda x: x.theme_index_inv if 'inv' in x.sign_to_theme_break else x.theme_index, axis=1)
    df['sign_to_theme_break'] = df.apply(lambda x: '-' if 'inv' in x.sign_to_theme_break else '+', axis=1)
    return df

#%% Merging H(S|C) = H(S) - MI 
def merge_model_contexts(models_df:pd.DataFrame, context_model:str, nocontext_model:str,
        merge_cols:list = ['file', 'index', 'speaker', 'text', 'theme','theme_index', 'theme_role'],
        int_cols:list = ['sum_h','normalised_h', 'xu_h', 'length'],
        context_extra_cols:list=[], nocontext_extra_cols:list=[], drop_na_nocontext:bool=True, 
        rename_patterns:list=['_ref', '_mod', '_diff']
    ) -> pd.DataFrame:
    print("Merging _ref (context: H(S|C)) and _mod (no context: H(S)) dataframes to compute MI (_diff) \nNote: MI = H(S) - H(S|C)")
    print("Warning: xu_h shouldn't be used for MI since sentences with different lengths have different references")

    cols = merge_cols + int_cols
    ref = models_df[models_df.model == context_model][cols+context_extra_cols]
    mod = models_df[models_df.model == nocontext_model][cols+nocontext_extra_cols]
    # merge
    c = pd.merge(left=ref, right=mod, left_on=merge_cols, right_on=merge_cols, 
            suffixes=(rename_patterns[0], rename_patterns[1]))
    if drop_na_nocontext:
        c.dropna(axis=0, subset=[f'xu_h{rename_patterns[1]}'], inplace=True)
    # adding diff columns
    for col in int_cols:
        c[f'{col}{rename_patterns[2]}'] = (c[f'{col}{rename_patterns[1]}'] - c[f'{col}{rename_patterns[0]}'])
        c[f'{col}_rel'] = c[f'{col}{rename_patterns[2]}'].abs()/c[f'{col}{rename_patterns[0]}']

    return c


#%% Window computations

def compute_window_pred_truth(df:pd.DataFrame, truth_col:str='truth_topic_change', 
                pred_col:str='pred_topic_change', windows:list=[2,5,10,15], print_tqdm:bool=False):
    # truth_col and pred_col must be boolean, 'change is here !!' kinda columns
    for window in windows:
        df[f"{truth_col}_win{window}"] = False
        df[f"{pred_col}_win{window}"] = False
        loop_on = tqdm(df.file.unique()) if print_tqdm else df.file.unique()
        for file in loop_on:
            idx = (df.file == file)
            df.loc[idx, f"{truth_col}_win{window}"] = df[idx][truth_col].rolling(window, center=True).apply(any)
            df.loc[idx, f"{pred_col}_win{window}"] = df[idx][pred_col].rolling(window, center=True).apply(any)
    return df

def compute_metrics(df:pd.DataFrame, truth_col:str='truth_topic_change', 
                pred_col:str='pred_topic_change', windows:list=[2,5,10,15]):
    d = {'precision':{}, 'recall':{}, 'cohen_kappa_score':{}} # precision > FP / recall > FN
    change_true = df[df[truth_col]].shape[0]
    change_pred = df[df[pred_col]].shape[0]

    d['cohen_kappa_score'][0] = cohen_kappa_score(df[truth_col], df[pred_col])
    tmp = df[df[pred_col]][truth_col]
    d['precision'][0] = tmp.sum()/change_pred
    tmp = df[df[truth_col]][pred_col]
    d['recall'][0] = tmp.sum()/change_true

    for window in windows:
        #tmp = df[df[f'{pred_col}_win{window}'].astype(bool).fillna(True)][f'{truth_col}_win{window}'] 
        # should only check if prediction 'in a window' matches a theme, but checking pred_window gives too many results
        d['cohen_kappa_score'][window] = cohen_kappa_score(df[f'{truth_col}_win{window}'].fillna(0.), df[f'{pred_col}_win{window}'].fillna(0.))
        tmp = df[df[pred_col].astype(bool).fillna(True)][f'{truth_col}_win{window}']
        d['precision'][window] = tmp.sum()/change_pred
        #tmp = df[df[f'{truth_col}_win{window}'].astype(bool).fillna(True)][f'{pred_col}_win{window}']
        tmp = df[df[truth_col].astype(bool).fillna(True)][f'{pred_col}_win{window}']
        d['recall'][window] = tmp.sum()/change_true
    
    d = pd.DataFrame(d)
    d['nb_true'] = change_true
    d['nb_pred'] = change_pred
    d['TP'] = change_true*d['recall']
    d['FP'] = change_pred*(1-d['precision'])
    d['FN'] = change_true*(1-d['recall'])
    d['f1_score'] = 2*(d['precision'] * d['recall']) / (d['precision'] + d['recall'])
    return d

#%% Compute outliers

def compute_outliers_skl(t:pd.DataFrame, col:str, n_neighbors:int=5, theme_role_col:str='theme_role'):
    # method 1: using apply
    #t['detrend_all'] = t.groupby('file').agg({col: lambda x: list(detrend(x))}).explode(col)[col].tolist() # values are in order
    # must sort for the other one
    #t.sort_values(by=['file','speaker','index'], inplace=True)
    #t['detrend_speaker'] = t.groupby(['file','speaker']).agg({col: lambda x: list(detrend(x))}).explode(col)[col].tolist()
    #t.sort_values(by=['file','index'], inplace=True)

    # method 2: using groups to compute outliers
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    sc = RobustScaler()
    t['detrend_all'] = None
    t['detrend_speaker'] = None
    t['scale_speaker'] = None
    t['gen_outlier'] = None
    t['speaker_outlier'] = None
    for file in t.file.unique():
        idx = (t.file == file).index
        spks = t.loc[idx].speaker.unique()
        t.loc[idx,"detrend_all"] = detrend(t.loc[idx,col])
        t.loc[idx,'gen_outlier'] = (clf.fit_predict(t.loc[idx,['index','detrend_all']]) == -1)
        for speaker in spks:
            idx = ((t.file == file) & (t.speaker == speaker)).index
            t.loc[idx,"detrend_speaker"] = detrend(t.loc[idx,col])
            t.loc[idx, "scale_speaker"] = sc.fit_transform(t.loc[idx,["detrend_speaker"]])
            t.loc[idx,'speaker_outlier'] = (clf.fit_predict(t.loc[idx,['index','detrend_speaker']]) == -1)

    t['initiator_outlier'] = t.speaker_outlier & (t[theme_role_col] == 'g')
    return t

#%% Plot outliers and themes
def plot_themes(t:pd.DataFrame, col:str='normalised_h', col_wrap:int=2, is_nested:bool=True, 
                    theme_col:str="theme", index_col:str="index", speaker_col:str="speaker", file_col:str="file", **kwargs):
    def _plot_themes(index, themes, **kwargs):
        tmp = pd.concat([themes, index], axis=1)
        tmp.columns = ['theme','index']
        new_themes = tmp.drop_duplicates(subset = "theme")["index"].tolist()
        for i in new_themes:
            plt.axvline(x=i, linestyle='dashed')

    spk_order = t.drop_duplicates([file_col,speaker_col])
    spk_order[index_col] = spk_order.groupby(file_col).agg({speaker_col: lambda x: range(len(x))})[speaker_col].explode().tolist()
    hue_order = spk_order.sort_values([index_col,file_col])[speaker_col].tolist()

    g = sns.relplot(
        data=t,
        x=index_col, y=col,
        hue=speaker_col, col=file_col, col_wrap=col_wrap, hue_order=hue_order,
        kind="line", 
        height=5, aspect=1.5, facet_kws=dict(sharex=False),
    )
    g.map(_plot_themes, index_col, theme_col)

    if is_nested:
        return g


def plot_themes_and_peaks(t:pd.DataFrame, col:str='normalised_h', outliers_source:str='gen', index_col:str="index", **theme_kwargs):
    def scatter_outliers(idx, vals, is_chosen, **kwargs):
        plt.scatter(idx[is_chosen], vals[is_chosen])
    
    g = plot_themes(t, col=col, **theme_kwargs)
    g.map(scatter_outliers, index_col, col, f"{outliers_source}_outlier")
    

#%% Aggregate to Word level functions
def concat_tokenized_to_words(s:str, word_tokens:list, entropy_tokens:list):
    """From the list of tokens & their entropy, reconstruct sentence word by word with associated word entropy
    """
    if len(word_tokens) == 0:
        return []
    elif isinstance(word_tokens, str):
        #print(s, word_tokens, entropy_tokens, type(word_tokens), type(entropy_tokens))
        #raise Exception
        #raise TypeError("arguments of type str, please cast beforehand.")
        word_tokens = literal_eval(word_tokens)
        entropy_tokens = literal_eval(entropy_tokens)
        # TODO: check srilm and <unk>, -inf

    #specific in case of the first word not being taken into account by the algorithm
    wt_1 = word_tokens[0]
    if wt_1 != s[:len(wt_1)]:
        s_start = re.search(wt_1,s).start()
        s = s[s_start:]
    
    # then run normally
    s = [x if i == 0 else ' '+x for i,x in enumerate(s.split(' '))] # WARNING: splitting removes space, but space is included in token 
    c = [] # storage

    current_word = ''
    current_ent = []
    ref_word = s.pop(0)
    for i, (tok, ent_tok) in enumerate(zip(word_tokens, entropy_tokens)):
        current_word += tok
        current_ent.append(ent_tok)

        #print(current_word)
        if current_word == ref_word:
            c.append({
                'word': current_word, 'word_sum_h':sum(current_ent), 'nb_tokens':len(current_ent), 
                'word_avg_h': sum(current_ent)/len(current_ent), 'ent_tokens':current_ent, 
                'sent_index': len(c)
            })
            # reset values
            current_word = ''
            current_ent = []
            if len(s) > 0:
                ref_word = s.pop(0)

        # could be adapted for words with issues matching

    return c

def word_level_explode(df:pd.DataFrame, keep_columns:list = ['corpus', 'file', 'dyad', 'index', 'speaker', 
                                            'start', 'stop', 'text', 'theme', 'theme_role', 'theme_index', 
                                            'has_theme', 'fb_type', 'sum_h', 'normalised_h', 'xu_h']) -> pd.DataFrame:
    tmp = df.copy(deep=True)
    tmp['entropy_word_level'] = tmp.apply(lambda x: concat_tokenized_to_words(x.text, x.tokens, x.tokens_h), axis=1)

    # Lines with errors - sentences with the '�' token cannot be matched
    errors = tmp[tmp.text.apply(lambda x: len(x.split(' '))) != tmp.entropy_word_level.apply(len)]
    print("Number of lines with errors: ", errors.shape[0])

    # Exploding then to Series then concat
    tmp = tmp.explode('entropy_word_level')
    print("Shape after exploding: ", tmp.shape)
    ewl_cols = list(tmp.iloc[0]['entropy_word_level'].keys()) # getting column names from conct_tokenized_to_words function
    tmp = pd.concat([tmp[keep_columns], tmp['entropy_word_level'].apply(pd.Series)], axis=1, ignore_index=True).reset_index(drop=True)
    tmp.columns = keep_columns + ewl_cols # renaming columns since ignore_index removed names
    tmp['ent_weight'] = tmp.word_sum_h.abs() / tmp.sum_h # cannot use word_avg_h since average, doesn't count as much for total sum

    return tmp, errors[['file','index', 'text', 'tokens', 'tokens_h']]

def word_level_sort(df:pd.DataFrame, word_importance_threshold:float=0.5,
        order_columns:list=['corpus', 'file', 'dyad', 'index', 'speaker', 'text'],
        extra_keep_columns:list=['theme', 'theme_role', 'theme_index', 'has_theme', 'fb_type', 'sum_h', 'normalised_h', 'xu_h']
    ) -> pd.DataFrame:
    """Note: groupby columns cannot contain nans """
    # Sorting for each sentence then cumusuming 
    tmp = df.sort_values(order_columns+['ent_weight'], ascending=[True]*len(order_columns)+[False])
    tmp['ent_weight_cs'] = tmp.groupby(order_columns)['ent_weight'].transform(pd.Series.cumsum)
    # Filtering by threshold
    tmp['ent_weight_filter'] = tmp.groupby(order_columns)['ent_weight_cs']\
        .transform('shift').fillna(0.) <= word_importance_threshold
    # Creating groupbyed version 
    tmp = tmp[tmp.ent_weight_filter]
    tmp_gp = tmp.groupby(order_columns).agg(
        dict({col: (lambda x: list(x)[0]) for col in extra_keep_columns}, 
        **{'word': list, 'word_avg_h':list, 'ent_weight': list})
    ).reset_index(drop=False)
    # Returning
    return tmp, tmp_gp