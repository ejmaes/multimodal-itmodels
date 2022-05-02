import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, re
from tqdm import tqdm

from collections import Counter
from itertools import chain
import logging
import argparse
import time

from sklearn.model_selection import train_test_split

#%% Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--original_dataframe", '-d', type=str, required=True, help="The path to the dataframe input to the SRILM model.")
    parser.add_argument("--slrim_output", '-f', type=str, default=None, help="The path to the input SRILM file.")
    parser.add_argument("--srilm_lm", '-lm', type=str, default=None, help="The path to the SRILM model.")
    parser.add_argument("--srilm_path", type=str, default=None, help="The path to the SRILM model.") #~/Downloads/srilm-1.7.3/bin/macosx/ngram
    parser.add_argument("--special_columns", '-c', type=dict, default={}, help="Original dataset interest columns - only add those varying from default {'text_col': 'text', 'file_col':'file'}.")
    parser.add_argument("--tts_seed", '-s', type=int, default=None, help="numpy seed if the original_dataframe needs train-test-split.")
    parser.add_argument("--output_token_df", action="store_true", help="Whether to explode the resulting df to token level.")

    args = parser.parse_args()
    # Check: either an output file, or a model and a file to apply SRILM on.
    if args.slrim_output is None and args.srilm_lm is None:
        raise argparse.ArgumentError('Cannot leave both srilm_lm and srilm_output empty, must specify one of them')
    if args.srilm_lm is not None and args.srilm_path is None:
        raise argparse.ArgumentError('To use SRILM must specify path')
    # Columns
    # Also for dataframe: column settings
    dataframe_col_default = {
        'text_col':'text', 'file_col':'file', 'index_col':'index', 'speaker_col':'speaker'
    }
    for k,v in dataframe_col_default.items():
        if k not in args.special_columns:
            args.special_columns[k] = v

    return args

def logger_setup():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO #if args.local_rank in [-1, 0] else logging.WARN,
    )
    return logger

#%% Parse SRILM file
"""
## Parse LM results into csv
SRILM parameters:
```
# Create LM
./ngram-count -text TEXTFILE -lm LM_NAME -order 3 %% -init-lm WIKPEDIA
# Execute LM
./ngram -lm LM -ppl FILE %% -no-sos -no-eos -unk -debug 2 other arguments
```
Documentation:
* http://www.speech.sri.com/projects/srilm/manpages/ngram-count.1.html
* http://www.speech.sri.com/projects/srilm/manpages/ngram.1.html


Typical output of the model:
```
on l'assimile
	p( on | <s> ) 	= [2gram] 0.003156502 [ -2.500794 ]
	p( l'assimile | on ...) 	= [2gram] 0.02406833 [ -1.618554 ]
	p( </s> | l'assimile ...) 	= [2gram] 0.1925466 [ -0.7154642 ]
1 sentences, 2 words, 0 OOVs
0 zeroprobs, logprob= -4.834812 ppl= 40.88877 ppl1= 261.4605

```
"""
def parse_srilmfile(lines:list):
    """Read the list of lines in input and parse into json with data for each line
    """
    stat_1 = "(\d+) sentences, (\d+) words, (\d+) OOVs"
    stat_2 = "(\d+) zeroprobs, logprob= ([+-]?(?:\d*\.\d+|\d+)) ppl= ((?:\d*\.\d+|\d+)) ppl1= ((?:\d*\.\d+|\d+))"
    stat_undef = "(\d+) zeroprobs, logprob= (.+) ppl= (.+) ppl1= (.+)"
    sent_proba = r"p\( (.+) \| (.+)\) = \[(\d)gram\] (.+) \[ (.+) \]"
    oov_proba = r"p\( (.+) \| (.+)\) = \[(.+)\] (.+) \[ (.+) \]"

    l = []
    d = {}
    for s in tqdm(lines[:-2]):
        s = s.translate(str.maketrans('', '', "\n\t")).strip()
        if s == '': # skip to next, n+1 line
            l.append(d)
            d = {}
        elif s[:3] == "p( ": # second to n-2 lines
            if 'OOV' not in s:
                w, cont, ngram, p, logp = re.match(sent_proba, s).groups()
                ngram = int(ngram)
            else:
                w, cont, ngram, p, logp = re.match(oov_proba, s).groups()
                ngram = np.nan
            d_words.append({
                'pred_word':w,
                'context':cont.strip(),
                'ngram_used': ngram,
                'p': float(p), 'logp': float(logp)
            })
        elif "OOVs" in s: # n-1 line
            ns, nw, noov = re.match(stat_1, s).groups()
            d = dict(d, **{'length': int(nw), 'nb_oov': int(noov), 'pred_tokens': d_words})
        elif "zeroprobs, logprob=" in s: # nth line
            m = re.match(stat_2, s)
            if m is not None:
                zp, logp, ppl, ppl1 = m.groups()
                d = dict(d, **{'sum_logp': float(logp), 'perplexity': float(ppl), 'perplexity_no-eos': float(ppl1)})
            else:
                zp, logp, ppl, ppl1 = re.match(stat_undef, s).groups()
                d = dict(d, **{'sum_logp': float(logp), 'perplexity': np.nan, 'perplexity_no-eos': np.nan})
        else: # first line
            d['test_text'] = s
            d['sent_index'] = len(l)
            d_words = []
    
    # last two lines are stats on the file
    s = lines[-2].translate(str.maketrans('', '', "\n\t")).strip()
    ns, nw, noov = re.search(stat_1, s).groups()
    s = lines[-1].translate(str.maketrans('', '', "\n\t")).strip()
    zp, logp, ppl, ppl1 = re.match(stat_2, s).groups()
    file_stats = {
        'number_of_words': int(nw), 'number_oov': int(noov),
        'file_logp': float(logp), 'file_perplexity': float(ppl), 'file_perplexity_no-eos': float(ppl1)
    }

    return l, file_stats



#%% Main
if __name__ == '__main__':
    args = parse_arguments()
    #logger = logger_setup()
    
    file_col = args.special_columns['file_col']
    text_col = args.special_columns['text_col']

    print('Loading DataFrame...')
    df = pd.read_csv(args.original_dataframe, keep_default_na=False, na_values=[''])
    # if no test
    if args.tts_seed is not None:
        files = df[file_col].unique()
        files_train, files_test = train_test_split(files, random_state=args.tts_seed, test_size=0.3)
        test_df = df[df[file_col].isin(files_test)].reset_index(drop=True)
    else:
        test_df = df.copy()
    # Check whether need to apply model
    if args.slrim_output is None:
        print('Applying SRILM to test data...')
        test_file = f'~/Downloads/test-data-text.txt'
        test_df[text_col].to_csv(test_file, index=False, header=False)
        args.slrim_output = f"{args.original_dataframe.replace('.csv','')}_srilm-{args.srilm_lm.split('/')[-1]}.txt"
        srilm_command = f"./{args.srilm_path} -lm {args.srilm_lm} -ppl {test_file} -unk -debug 2 > {args.slrim_output}"
        print(f"SRILM command: `{srilm_command}`")
        os.system(srilm_command)
        args.srilm_lm = args.srilm_lm.split('/')[-1] # rename for later convenience in naming model
    
    # Parse file
    with open(args.slrim_output, 'r') as f:
        lines = f.readlines()
        print('Loading SRILM output file...')
        l, file_stats = parse_srilmfile(lines)

    l = pd.DataFrame(l)
    print(f'SRILM File stats: {file_stats} \nDataframe shape: {l.shape}')
    l['tokens'] = l['pred_tokens'].apply(lambda x: [d['pred_word'] for d in x if d['pred_word'] != '<s>'])
    l['tokens_h'] = l['pred_tokens'].apply(lambda x: [d['logp'] for d in x if d['pred_word'] != '<s>'])
    # Merge with original
    if l.shape[0] != test_df.shape[0]:
        #raise IndexError("DataFrames don't have the same shape, merging not possible")
        print("DataFrames don't have the same shape, merging not possible -- exporting data without merging")
        c = l.copy()
    else:
        c = pd.concat([test_df, l], axis=1)
        # Check merged properly
        #assert (c['test_text'].apply(lambda x: '' if str(x) == 'nan' else x.strip().lower()) != c[text_col].apply(lambda x: '' if str(x) == 'nan' else x.strip().lower())).sum() == 0
        # If OK save the df
    model_name = 'srilm_'+args.slrim_output.split('/')[-1].replace('.txt','')
    c['model'] = f"srilm-{args.srilm_lm if args.srilm_lm is not None else ''}"
    c.to_csv(args.slrim_output.replace('.txt','.csv'), index=False)
    
    # Focus on tokens:
    if args.output_token_df:
        #df_tokens = pd.concat(l.pred_tokens.apply(pd.DataFrame).tolist(), axis=0).reset_index(drop=True)
        df_tokens = pd.concat( c.pred_tokens.apply(pd.DataFrame).tolist(), # exploding list of tokens/perplexity
                        keys=c[['file','index', 'speaker','text']].apply(tuple, axis=1).tolist() # adding index from file etc
                    ).reset_index().rename(columns={f'level_{i}':k for i,k in enumerate(['file','index', 'speaker','text', 'word_index'])}) # rename index columns
        print(f"Number of tokens:{df_tokens.shape[0]} \nDistribution of n-gram used: {df_tokens.ngram_used.value_counts()}")
        df_tokens.to_csv(args.slrim_output.replace('.txt','-tokens.csv'), index=False)