# %% Adapted from Xu & Reitter 2017, acl2017-spectral-master
from xml.dom import IndexSizeErr
import pandas as pd
import numpy as np
import os,sys,re
from glob import glob
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functools import partial
from scipy.interpolate import interp1d
import statsmodels.api as sm

import argparse

# %%
# python spectrum R lib
SPEC_PATH = "/Users/neako/Documents/Cours-MasCo/PhD/tools/githubs/IT/time-series-analysis-master/Python"
sys.path.append(SPEC_PATH)
from spectrum import * # spec_pgram


#%% Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", '-d', type=str, required=True, help="The path to the input pandas dataframe.")
    parser.add_argument("--aggregate", '-g', action='store_true', help="Whether there is a need to agregate adjacent rows from the same speaker")
    parser.add_argument("--theme_apply", '-t', action='store_true', help="Whether to apply to themes instead of files")
    parser.add_argument("--min_pause", '-p', type=float, default=0.3, help="Minimum duration in seconds between two IPUs, if aggregating; default 0.3s")
    parser.add_argument("--moves_and_deviation", '-da', action="store_true", help="Additional data for performance analysis")
    parser.add_argument("--dataframe_col_special", '-dfc', type=dict, default={}, help=f"Dataframe columns - if empty keep default.")
    parser.add_argument("--file_model", type=str, default=None, help=f"If several models in the file, which file to keep - model_col must be set in dataframe_col_special")
    parser.add_argument("--fft_on", '-f', type=str, default='xu_h', help=f"Which column (xu_h, normalised_h) to run analysis on.")

    args = parser.parse_args()
    # Also for dataframe: column settings
    dataframe_col_default = {
        'text_col':'text', 'file_col':'file', 'index_col':'index', 'theme_col':'theme', 'speaker_col':'speaker', 'xu_col':'xu_h',
        'tokens_col':'tokens', 'sum_h_col':'sum_h', 'length_col':'length', 'normalised_h_col':'normalised_h'
    }
    for k,v in dataframe_col_default.items():
        if k not in args.dataframe_col_special:
            args.dataframe_col_special[k] = v

    if not os.path.exists(args.data_path):
        raise FileExistsError(f'Data file {args.data_path} not found.')
    if args.theme_apply and 'theme_col' not in args.dataframe_col_special:
        raise argparse.ArgumentError('--theme_apply requires for theme_role and theme_index to be set in --dataframe_col_special {}')
    if args.file_model is not None and 'model_col' not in args.dataframe_col_special:
        raise argparse.ArgumentError('--file_model requires for model_col to be set in --dataframe_col_special {}')

    return args

def aggregate_ipus(dataframe:pd.DataFrame, min_pause_duration:float = 0.3, 
                    text_col:str='text', file_col:str='file', speaker_col:str='speaker',
                    tokens_col:str='tokens', sum_h_col:str='sum_h', length_col:str='length', normalised_h_col:str="normalised_h", xu_h_col:str="xu_h", **kwargs):
    df = dataframe.copy(deep=True)
    df['cs'] = ((df[file_col] != df[file_col].shift()) | (df[speaker_col] != df[speaker_col].shift())).cumsum()
    
    df.groupby([file_col,'cs']).agg({
        text_col: ' '.join, speaker_col: lambda x: list(x)[0],
        tokens_col: lambda x: itertools.chain(*x), sum_h_col: 'sum', length_col: 'sum'
    }).reset_index(drop=False)
    df[normalised_h_col] = df[sum_h_col] / df[length_col]
    h_bar = dataframe.groupby(length_col).agg({normalised_h_col: "mean"}).to_dict()[normalised_h_col]
    dataframe[xu_h_col] = dataframe.apply(lambda x: np.nan if x[length_col] not in h_bar else x[normalised_h_col]/h_bar[x[length_col]], axis=1)

    return df

# %% Power Spectrum Overlap
def compute_pso(df_ent_pso:pd.DataFrame, file_col:str="file", speaker_col:str="speaker", **kwargs):
    pso = []
    for f in df_ent_pso[file_col].unique(): # was df
        df_ent_sel = df_ent_pso[df_ent_pso[file_col] == f] # was df_ent
        speakers = df_ent_sel[speaker_col].unique()
        if len(speakers) != 2:
            raise IndexSizeErr(f"Should only be two speakers in file {f}")
        elif set(speakers) == set(['f','g']):
            f = 'f'
            g = 'g'
        else:
            f = speakers[0]
            g = speakers[1]
        x_g = df_ent_sel[df_ent_sel[speaker_col] == g].freq
        y_g = df_ent_sel[df_ent_sel[speaker_col] == g].spec
        x_f = df_ent_sel[df_ent_sel[speaker_col] == f].freq
        y_f = df_ent_sel[df_ent_sel[speaker_col] == f].spec
        # linear interpolation - slightly modified to accompodate python interp1d which throws error instead of NA for values not in x
        x_out = pd.concat([x_g, x_f]).astype(float).sort_values().unique()
        x_out_g = x_out[np.where((x_out >= x_g.min()) & (x_out <= x_g.max()))[0]]
        x_out_f = x_out[np.where((x_out >= x_f.min()) & (x_out <= x_f.max()))[0]]
        # CANNOT USE THOSE DIRECTLY since shape differ
        y_out_f = interp1d(x_f, y_f)(x_out_f) # aka approx_[f,g]
        y_out_g = interp1d(x_g, y_g)(x_out_g) # approx$y est la valeur de np.intrep
        # ADDING NANs
        approx_f = interp1d(x_f, y_f) 
        approx_g = interp1d(x_g, y_g)
        approx_f = np.array([approx_f(x) if x in x_out_f else np.nan for x in x_out]) # contains out values
        approx_g = np.array([approx_g(x) if x in x_out_g else np.nan for x in x_out]) # contains out values
        # NOTE: np.interp(x_out, x_f, y_f) also works but doesn't add nans, adds 1rst non nan value in place
        
        # find min ys and remove NAs
        y_min = np.minimum(approx_f, approx_g)
        x_min = x_out[~np.isnan(y_min)]
        y_min = y_min[~np.isnan(y_min)]
        y_max = np.maximum(approx_f, approx_g)
        x_max = x_out[~np.isnan(y_max)]
        y_max = y_max[~np.isnan(y_max)]

        # compute AUVs and PSO
        AUV_g = np.trapz(y_out_g, x_out_g)  # y,x
        AUV_f = np.trapz(y_out_f, x_out_f)
        AUV_min = np.trapz(y_min, x_min)
        AUV_max = np.trapz(y_max, x_max)
        # PSO = AUV_min / (AUV_g + AUV_f)
        PSO = AUV_min / AUV_max
        # return PSO
        pso.append({'file': f, 'PSO': PSO, 'AUVg': AUV_g, 'AUVf': AUV_f, 'AUVmin': AUV_min})

    pso = pd.DataFrame(pso)
    return pso


# %% RP
def compute_rp(dataframe:pd.DataFrame, speaker_col:str="speaker", xu_col:str="xu_h", file_col:str="file", **kwargs):
    df = dataframe.copy(deep=True)

    rp = []
    for f in df[file_col].unique():
        max_len = min(df[df[file_col] == f].groupby(speaker_col).count()[xu_col].tolist())
        y_f = df[(df[file_col] == f) & (df[speaker_col] == 'f')][xu_col].iloc[:max_len]
        y_g = df[(df[file_col] == f) & (df[speaker_col] == 'g')][xu_col].iloc[:max_len]
        comb_ts = pd.concat([y_f.reset_index(drop=True), y_g.reset_index(drop=True)], axis=1) #, ignore_index=True - not working
        spec = spec_pgram(comb_ts, taper=0, plot=False, log='no', detrend=False)
        spec_df = pd.DataFrame(spec['spec'])

        # phase shift at all peaks
        i_max = np.sign(spec_df.diff()).diff().apply(lambda x: np.where(x < 0)[0], axis=0)
        i_index = np.array(list(set(np.concatenate((i_max[0],i_max[1]))))) - 1
        peakPS = spec['phase'].T[0][i_index]
        rp.append({'file': f, 'peakPS': peakPS})

    rp = pd.DataFrame(rp).explode('peakPS')
    return rp



if __name__ == '__main__':
    args = parse_arguments()
    df = pd.read_csv(args.data_path)
    save_file_pat = f"{args.data_path.replace('.csv','')}_agg{int(args.aggregate)}_theme{int(args.theme_apply)}{f'_{args.file_model}' if args.file_model is not None else ''}_{args.fft_on}"

    if args.file_model is not None:
        df = df[df[args.dataframe_col_special['model_col']] == args.file_model]
    if args.aggregate:
        df = aggregate_ipus(df, min_pause_duration=args.min_pause, **args.dataframe_col_special)
    if args.theme_apply:
        args.dataframe_col_special['speaker_col'] = args.dataframe_col_special['theme_role']
        args.dataframe_col_special['index_col'] = args.dataframe_col_special['theme_index']
        df = df[(~df[args.dataframe_col_special['theme_col'].isna()]) & (df[args.dataframe_col_special['theme_col'] != " "])]
        df['filextheme'] = df[args.dataframe_col_special['file_col']] + ' x ' + df[args.dataframe_col_special['theme_col']]
        args.dataframe_col_special['file_col'] = 'filextheme'
    args.dataframe_col_special['xu_h'] = args.fft_on

    # Compute PSO
    # spec_pgram is a function from spectrum
    file_col = args.dataframe_col_special['file_col']
    xu_col = args.dataframe_col_special['xu_h']
    speaker_col = args.dataframe_col_special['speaker_col']
    index_col = args.dataframe_col_special['index_col']
    df = df.sort_values([file_col,index_col])

    spgram = partial(spec_pgram, taper=0, plot=False, log='no')
    df_ent = df.groupby([file_col,speaker_col]).agg({xu_col:spgram})
    df_ent_pso = df_ent[xu_col].apply(pd.Series)[['freq','spec']].explode(['freq','spec']).reset_index(drop=False)
    df_ent_pso.freq = df_ent_pso.freq.astype(float)
    df_ent_pso.spec = df_ent_pso.spec.astype(float)

    pso = compute_pso(df_ent_pso)

    # Compute RP
    rp = compute_rp(df)

    df_to_save = {"_rp":rp, "_pso":pso, "_fft": df_ent_pso}

    # %% [markdown]
    # Load extra data such as pathdev from `'data/moves_and_deviation.csv'` for modeling and plotting
    if args.moves_and_deviation:
        df_path = pd.read_csv(args.moves_and_deviation)
        psom = pd.merge(left = pso, right = df_path, left_on="file", right_on="Observation")

        # %% [markdown]
        # How to add intercepts: https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
        model = sm.OLS(psom['path dev'], psom['PSO']).fit() ## sm.OLS(output, input)
        model.summary()
    
        sns.jointplot(x="PSO", y="path dev", data=psom, kind="reg")

        # RP
        df_path_match = df_path.set_index('Observation')['path dev'].to_dict()
        rp['pathdev'] = rp[file_col].apply(lambda x: np.nan if x not in df_path_match else df_path_match[x])
        rp[~np.isnan(rp.pathdev)] # OK shape

        # Compute mean, median and max values for peakPS
        rp_mean = rp.groupby('file').agg({
            'pathdev':'mean', 
            'peakPS': [lambda x: np.mean(np.abs(x)), lambda x: np.median(np.abs(x)), lambda x: np.max(np.abs(x))]
        }).droplevel(0, axis=1)
        rp_mean.columns = ['pathdev', 'mean', 'median', 'max']
        df_to_save["_rp_mean":rp_mean]

    for k,v in df_to_save.items():
        v.to_csv(f'{save_file_pat}{k}.csv', index=False)
